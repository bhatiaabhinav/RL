import cplex  # noqa: F401
import numpy as np
from RL.common.utils import kdel


class OptnetLayer:
    def __init__(self, n_outputs, n_inputs):
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs

    def forward_pass(self, input_vector):
        raise NotImplementedError()

    def gradients(self):
        '''
        returns gradients (Jacobian matrix) of the current output w.r.t current input
        by the method of differentiating the KKT conditions at the optimal point w.r.t the input variables
        '''
        raise NotImplementedError()


class ConstrainedProjectionOptnetLayer(OptnetLayer):
    def __init__(self, n_outputs, n_inputs, constraints):
        super().__init__(n_outputs, n_inputs)
        self.constraints = constraints

    def _cplex_nearest_feasible_sum_min_max_constraints(self, vec_y, c, vec_cmin, vec_cmax):
        """
        solves L.P 1 from document ddpg-cp-optnet.pdf
        let d_i = |z_i - a_i|
        obj = min sum d_i
        subject to:
            sum z_i = C                     (lambda_)
            forall_i: -z_i - d_i <= -y_i    (mu_i)
            forall_i: z_i - d_i <= y_i      (nu_i)
            forall_i: z_i <= cmax_i         (alpha_i)
            forall_i: -z_i <= -cmin_i       (beta_i)
        where lambda_, mu_i, nu_i, alpha_i, beta_i are lagrange multipliers

        Arguments:
            vec_y -- inputs
            c -- sum constraint
            vec_cmin -- min constraints
            vec_cmax -- max constraints

        Returns:
            vec_z, vec_d, lambda_, vec_mu, vec_nu, vec_alpha, vec_beta, cplex_problem
        """
        vec_y = list(vec_y)
        k = len(vec_y)
        # names:
        z_names = ['z_' + str(i) for i in range(k)]
        d_names = ['d_' + str(i) for i in range(k)]

        prob = cplex.Cplex()

        # objective:
        prob.objective.set_sense(prob.objective.sense.minimize)
        names = z_names + d_names
        obj = [0] * k + [1] * k
        prob.variables.add(obj=obj, names=names)

        row_names = []
        lhss = []
        senses = []
        rhss = []

        # lambda_:
        row_names.append('lambda_')
        lhss.append([z_names, [1] * k])
        senses.append('E')
        rhss.append(c)

        # u_i:
        for i in range(k):
            row_names.append('mu_' + str(i))
            lhss.append([['z_' + str(i), 'd_' + str(i)], [-1, -1]])
            senses.append('L')
            rhss.append(-vec_y[i])

        # v_i:
        for i in range(k):
            row_names.append('nu_' + str(i))
            lhss.append([['z_' + str(i), 'd_' + str(i)], [1, -1]])
            senses.append('L')
            rhss.append(vec_y[i])

        # alpha_i:
        for i in range(k):
            row_names.append('alpha_' + str(i))
            lhss.append([['z_' + str(i)], [1]])
            senses.append('L')
            rhss.append(vec_cmax[i])

        # beta_i:
        for i in range(k):
            row_names.append('beta_' + str(i))
            lhss.append([['z_' + str(i)], [-1]])
            senses.append('L')
            rhss.append(-vec_cmin[i])

        assert len(row_names) == len(lhss) == len(senses) == len(rhss) == 4 * k + \
            1, "There should be 4k+1=4*{0}+1={1} constraints. But found {2}".format(
                k, 4 * k + 1, len(row_names))

        prob.linear_constraints.add(
            lin_expr=lhss, senses=senses, rhs=rhss, names=row_names)

        prob.set_results_stream(None)
        prob.set_log_stream(None)

        prob.solve()

        vec_z = np.array(prob.solution.get_values()[0:k])
        vec_d = np.array(prob.solution.get_values()[k:])
        dual_vars = prob.solution.get_dual_values()
        assert len(dual_vars) == 4 * k + \
            1, "Number of dual varaibles should have been same as no. of constaints"
        lambda_ = dual_vars[0]
        vec_mu = np.array(dual_vars[1:1 + k])
        vec_nu = np.array(dual_vars[1 + k:1 + 2 * k])
        vec_alpha = np.array(dual_vars[1 + 2 * k:1 + 3 * k])
        vec_beta = np.array(dual_vars[1 + 3 * k:])
        return vec_z, vec_d, lambda_, vec_mu, vec_nu, vec_alpha, vec_beta, prob

    def forward_pass(self, input_vector):
        print('right now only 1-level constraints are supported')
        c = self.constraints['equals']
        cmin = [child["min"] for child in self.constraints["children"]]
        cmax = [child["max"] for child in self.constraints["children"]]
        z, d, lamda_, mu, nu, alpha, beta, cplex_problem = self._cplex_nearest_feasible_sum_min_max_constraints(
            input_vector, c, cmin, cmax)
        cplex_problem.write('test.lp')
        self._saved_vars = (input_vector, c, cmin, cmax, z,
                            d, lamda_, mu, nu, alpha, beta)
        return z

    def gradients(self):
        '''
        returns gradients (Jacobian matrix) of the current output w.r.t current input
        by the method of differentiating the KKT conditions of the LP at the optimal point w.r.t the input variables
        '''
        y, c, cmin, cmax, z, d, lambda_, mu, nu, alpha, beta = self._saved_vars
        k = len(z)
        n = 6 * k + 1
        jacobian = np.zeros(shape=[k, k])
        for j in range(k):
            # section 5 of ddpg-cp-optnet.pdf:
            # this to calculate jacobian w.r.t y_j
            A = np.zeros(shape=[n, n])
            B = np.zeros(shape=[n])
            for m in range(1, k + 1):
                A[0, 6 * m] = 1
                A[6 * m - 5, 6 * m - 5] = A[6 * m - 5, 6 * m - 4] = 1
                A[6 * m - 4, 0] = A[6 * m - 4, 6 *
                                    m - 4] = A[6 * m - 4, 6 * m - 3] = 1
                A[6 * m - 4, 6 * m - 5] = A[6 * m - 4, 6 * m - 2] = -1
                A[6 * m - 3, 6 * m - 5] = y[m - 1] - z[m - 1] - d[m - 1]
                A[6 * m - 3, 6 * m - 1] = A[6 * m - 3, 6 * m] = -mu[m - 1]
                A[6 * m - 2, 6 * m - 4] = -y[m - 1] + z[m - 1] - d[m - 1]
                A[6 * m - 2, 6 * m - 1] = -nu[m - 1]
                A[6 * m - 2, 6 * m] = nu[m - 1]
                A[6 * m - 1, 6 * m - 3] = z[m - 1] - cmax[m - 1]
                A[6 * m - 1, 6 * m] = alpha[m - 1]
                A[6 * m, 6 * m - 2] = -z[m - 1] + cmin[m - 1]
                A[6 * m, 6 * m] = -beta[m - 1]

                B[6 * m - 3] = -mu[m - 1] * kdel(m - 1, j)
                B[6 * m - 2] = nu[m - 1] * kdel(m - 1, j)

            print("A:\n", A[145:151, 145:151])
            jacobian_yj = np.linalg.solve(A, B)
            # but we derivatives only of zs:
            for m in range(k):
                jacobian[m, j] = jacobian_yj[6 + 6 * m]

        # k x k gradients of z w.r.t y. grad[i][j]=dz_j/dy_i
        return np.transpose(jacobian)


class ConstrainedProjectionL2OptnetLayer(OptnetLayer):
    def __init__(self, n_outputs, n_inputs, constraints):
        super().__init__(n_outputs, n_inputs)
        self.constraints = constraints

    def _cplex_nearest_feasible_sum_min_max_constraints(self, vec_y, c, vec_cmin, vec_cmax):
        """
        solves QP 1 from document ddpg-cp-qp-optnet.pdf
        obj = min sum (z_i - y_i)^2
        subject to:
            sum z_i = C                     (lambda_)
            forall_i: z_i <= cmax_i         (alpha_i)
            forall_i: -z_i <= -cmin_i       (beta_i)
        where lambda_, alpha_i, beta_i are lagrange multipliers

        Arguments:
            vec_y -- inputs
            c -- sum constraint
            vec_cmin -- min constraints
            vec_cmax -- max constraints

        Returns:
            vec_z, lambda_, vec_alpha, vec_beta, cplex_problem
        """
        vec_y = list(vec_y)
        k = len(vec_y)
        # names:
        z_names = ['z_' + str(i) for i in range(k)]

        prob = cplex.Cplex()

        # objective:
        prob.objective.set_sense(prob.objective.sense.minimize)
        names = z_names
        obj = list(-2 * np.asarray(vec_y))
        prob.variables.add(obj=obj, names=names)
        # since Q=2I, i.e. a seperable quadratic objective, we can just give coeffs of z_i^2
        qmat = 2 * np.ones(k)
        prob.objective.set_quadratic(qmat)

        row_names = []
        lhss = []
        senses = []
        rhss = []

        # lambda_:
        row_names.append('lambda_')
        lhss.append([z_names, [1] * k])
        senses.append('E')
        rhss.append(c)

        # alpha_i:
        for i in range(k):
            row_names.append('alpha_' + str(i))
            lhss.append([['z_' + str(i)], [1]])
            senses.append('L')
            rhss.append(vec_cmax[i])

        # beta_i:
        for i in range(k):
            row_names.append('beta_' + str(i))
            lhss.append([['z_' + str(i)], [-1]])
            senses.append('L')
            rhss.append(-vec_cmin[i])

        assert len(row_names) == len(lhss) == len(senses) == len(rhss) == 2 * k + \
            1, "There should be 2k+1=2*{0}+1={1} constraints. But found {2}".format(
                k, 2 * k + 1, len(row_names))

        prob.linear_constraints.add(
            lin_expr=lhss, senses=senses, rhs=rhss, names=row_names)

        prob.set_results_stream(None)
        prob.set_log_stream(None)

        prob.solve()

        vec_z = np.array(prob.solution.get_values())
        dual_vars = prob.solution.get_dual_values()
        assert len(dual_vars) == 2 * k + \
            1, "Number of dual varaibles should have been same as no. of constaints"
        lambda_ = dual_vars[0]
        vec_alpha = np.array(dual_vars[1:1 + k])
        vec_beta = np.array(dual_vars[1 + k:1 + 2 * k])
        return vec_z, lambda_, vec_alpha, vec_beta, prob

    def forward_pass(self, input_vector):
        print('right now only 1-level constraints are supported')
        c = self.constraints['equals']
        cmin = [child["min"] for child in self.constraints["children"]]
        cmax = [child["max"] for child in self.constraints["children"]]
        # z, d, lamda_, mu, nu, alpha, beta, cplex_problem = cplex_nearest_feasible_sum_min_max_constraints(
        #     input_vector, c, cmin, cmax)
        # self._saved_vars = (input_vector, c, cmin, cmax, z,
        #                     d, lamda_, mu, nu, alpha, beta)
        z, lamda_, alpha, beta, cplex_problem = self._cplex_nearest_feasible_sum_min_max_constraints(
            input_vector, c, cmin, cmax)
        self._saved_vars = (input_vector, c, cmin, cmax,
                            z, lamda_, alpha, beta)

        cplex_problem.write('test.lp')
        return z

    def gradients(self):
        '''
        returns gradients (Jacobian matrix) of the current output w.r.t current input by the method of differentiating the KKT conditions of the LP at the optimal point w.r.t the input variables
        '''
        y, c, cmin, cmax, z, lamda_, alpha, beta = self._saved_vars
        k = len(z)
        n = 3 * k + 1
        jacobian = np.zeros(shape=[k, k])

        # section 5 of ddpg-cp-qp-optnet.pdf:
        # from eq 9, coeff matrix for all set of linear equations is same
        A = np.zeros(shape=[n, n])
        for m in range(1, k + 1):
            A[0, 3 * m] = 1
            A[3 * m - 2, 0] = 1
            A[3 * m - 2, 3 * m - 2] = 1
            A[3 * m - 2, 3 * m - 1] = -1
            A[3 * m - 2, 3 * m] = 2
            A[3 * m - 1, 3 * m - 2] = z[m - 1] - cmax[m - 1]
            A[3 * m - 1, 3 * m] = alpha[m - 1]
            A[3 * m, 3 * m - 1] = -z[m - 1] + cmin[m - 1]
            A[3 * m, 3 * m] = -beta[m - 1]

        print("A:\n", A[4:7, 4:7])
        A_inv = np.linalg.inv(A)

        for j in range(k):
            B_yj = np.zeros(shape=[n])
            B_yj[3 * j + 1] = 2

            jacobian_yj = np.matmul(A_inv, B_yj)
            # but we want to store derivatives only of zs:
            for m in range(k):
                jacobian[m, j] = jacobian_yj[3 + 3 * m]

        # k x k gradients of z w.r.t y. grad[i][j]=dz_j/dy_i
        return np.transpose(jacobian)


def test_optnet():
    from RL.reco_rl.wrappers import ERStoMMDPWrapper
    from RL.reco_rl.constraints import normalize_constraints
    import gym_ERSLE  # noqa: F401
    import gym
    import numpy as np
    env = gym.make('pyERSEnv-ca-dynamic-cap6-30-v6')
    env = ERStoMMDPWrapper(env)
    c = env.metadata['constraints']
    normalize_constraints(c)
    n = env.action_space.shape[0]
    optnet_layer = ConstrainedProjectionL2OptnetLayer(n, n, c)
    test_input = np.asarray(
        [8, 4, 3, 1, 7, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) / 32
    print(test_input * 32)
    feasible_action = optnet_layer.forward_pass(test_input)
    grads = optnet_layer.gradients()
    print('grads:\n', grads)
    print('feasible action:\n', feasible_action * 32)
    # if it is feasible action, doing another forward pass should not change it:
    feasible_new = optnet_layer.forward_pass(feasible_action)
    print('feasible action after another pass:\n', feasible_new * 32)
    assert np.all(feasible_action == feasible_new)


if __name__ == '__main__':
    test_optnet()
