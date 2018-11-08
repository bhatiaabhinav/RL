import time
from multiprocessing import Pool

import cplex  # noqa: F401
import numpy as np
import tensorflow as tf

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


def _cplex_nearest_feasible_sum_min_max_constraints(vec_y, c, vec_cmin, vec_cmax):
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
    obj = list(-2 * np.array(vec_y).astype(float))
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
    dual_vars = np.array(prob.solution.get_dual_values())
    assert len(dual_vars) == 2 * k + \
        1, "Number of dual varaibles should have been same as no. of constaints"
    lambda_ = dual_vars[0]
    vec_alpha = dual_vars[1:1 + k]
    vec_beta = dual_vars[1 + k:1 + 2 * k]
    return vec_z, lambda_, vec_alpha, vec_beta, prob


def forward_pass(input_vector, c, cmin, cmax):
    input_vector = input_vector.astype(float)
    c = float(c)
    cmin = cmin.astype(float)
    cmax = cmax.astype(float)
    if abs(np.sum(input_vector) - c) < 1e-6 and np.all(input_vector >= cmin) and np.all(input_vector <= cmax):
        z, lambda_, alpha, beta = input_vector, 0, np.zeros([len(cmin)]), np.zeros([len(cmax)])
        # print('savings')
    else:
        z, lambda_, alpha, beta, cplex_problem = _cplex_nearest_feasible_sum_min_max_constraints(
            input_vector, c, cmin, cmax)
        # cplex_problem.write('test.lp')
    return z, (lambda_, alpha, beta)


def batch_forward_pass(input_vectors, c, cmin, cmax):
    input_vectors = np.array(input_vectors)
    batch_size = np.shape(input_vectors)[0]
    z = np.zeros(np.shape(input_vectors), dtype=np.float32)
    for idx in range(batch_size):
        z[idx], duals = forward_pass(input_vectors[idx], c, cmin, cmax)
    return z


def batch_forward_pass_multithreaded(input_vectors, c, cmin, cmax):
    input_vectors = list(input_vectors)
    batch_size = len(input_vectors)
    pool = Pool()
    answers = pool.map(forward_pass, zip(
        input_vectors, [c] * batch_size, [cmin] * batch_size, [cmax] * batch_size))
    pool.close()
    pool.join()
    return np.array([z for z, duals in answers])


def gradients(input_vector, c, cmin, cmax):
    '''
    returns gradients (Jacobian matrix) of the current output w.r.t input by the method of differentiating the KKT conditions of the LP at the optimal point w.r.t the input variables
    '''
    # y = input_vector
    z, duals = forward_pass(input_vector, c, cmin, cmax)
    lamda_, alpha, beta = duals
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

    # print("A:\n", A[4:7, 4:7])
    # A += 1e-5
    A_inv = np.linalg.inv(A)

    for j in range(k):
        B_yj = np.zeros(shape=[n])
        B_yj[3 * j + 1] = 2

        jacobian_yj = np.matmul(A_inv, B_yj)
        # but we want to store derivatives only of zs:
        for m in range(k):
            jacobian[m, j] = jacobian_yj[3 + 3 * m]

    # k x k gradients of z w.r.t y. grad[i][j]=dz_i/dy_j
    return jacobian  # z x y


def batch_gradients(input_vectors, c, cmin, cmax):
    input_vectors = np.array(input_vectors)
    batch_size = np.shape(input_vectors)[0]
    jacobians = []
    for idx in range(batch_size):
        # z x y matrix
        jacobians.append(gradients(input_vectors[idx], c, cmin, cmax))
    return np.array(jacobians).astype(np.float32)


# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def _tf_constrained_project_grad(op, grads_wrt_z):
    print("gradient graph being set up")
    # grads_wrt_z is of shape [batch_size, len(z)] = [batch_size, k]
    grads_wrt_z = tf.expand_dims(grads_wrt_z, 1)  # [bs, 1, len(z)]
    ys = op.inputs[0]  # [batch_size, k]
    c = op.inputs[1]  # [1]
    cmin = op.inputs[2]  # [k]
    cmax = op.inputs[3]  # [k]
    k = cmin.get_shape().as_list()[0]
    print(k)
    # [batch_size, len(z), len(y)] = [batch_size, k, k]
    jacobians_z_wrt_y = tf.py_func(batch_gradients, [ys, c, cmin, cmax], tf.float32, name='batch_gradients')
    grads_wrt_y = tf.reshape(tf.matmul(grads_wrt_z, jacobians_z_wrt_y), [-1, k])  # will give [bs, k]
    grads_wrt_c = tf.constant(0)
    grads_wrt_cmin = tf.zeros([k])
    grads_wrt_cmax = tf.zeros([k])
    return grads_wrt_y, grads_wrt_c, grads_wrt_cmin, grads_wrt_cmax


def tf_constrained_project_nearest_L2(y, c, cmin, cmax, name='constrained_project_nearest_L2'):
    with tf.name_scope(name, "Constrained_Projection_L2", [y, c, cmin, cmax]) as scope:
        ans = py_func(batch_forward_pass, [y, c, cmin, cmax], tf.float32, name=scope, grad=_tf_constrained_project_grad)
        ans = tf.reshape(ans, [-1, len(cmin)])
        return ans


def timeit(f, repeat=1000):
    t_start = time.time()
    for i in range(repeat):
        ans = f()
    t_end = time.time()
    fps = repeat / (t_end - t_start)
    print('fps', fps)
    return ans


def test_optnet():
    from RL.reco_rl.wrappers import ERStoMMDPWrapper
    from RL.reco_rl.constraints import normalize_constraints
    import gym_ERSLE  # noqa: F401
    import gym
    import numpy as np
    env = gym.make('pyERSEnv-ca-dynamic-cap6-30-v6')
    env = ERStoMMDPWrapper(env)
    cons = env.metadata['constraints']
    normalize_constraints(cons)
    c = cons['equals']
    cmin = np.array([child["min"] for child in cons["children"]])
    cmax = np.array([child["max"] for child in cons["children"]])
    # optnet_layer = ConstrainedProjectionL2OptnetLayer(n, n, c)
    # test_input = np.asarray([5.5, 4, 3, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0]) / 32
    ti_3 = np.array([8] * 12 + [7] * 13) / 32
    # print(test_input * 32)
    # feasible_action, duals = forward_pass(test_input, c, cmin, cmax)
    # grads = gradients(test_input, c, cmin, cmax)
    # np.set_printoptions(precision=2)
    # print('feasible action:\n', feasible_action * 32)
    # print('grads:\n', np.round(grads, 3))
    # # if it is feasible action, doing another forward pass should not change it:
    # feasible_new, duals = forward_pass(feasible_action, c, cmin, cmax)
    # print('feasible action after another pass:\n', feasible_new * 32)
    # # when the action is feasible, expect jacobian to be identity
    # grads = gradients(feasible_action, c, cmin, cmax)
    # print('gradients at a feasible action should be identity:\n', np.round(grads, 3))
    # # test_input2 = np.asarray([0] * 25) / 32
    # # feasible_action, duals = forward_pass(test_input2, c, cmin, cmax)
    # # grads = gradients(test_input2, c, cmin, cmax)
    # print('feasible action:\n', feasible_action * 32)
    # print('grads:\n', np.round(grads, 3))

    # tf_inp_feed = tf.placeholder(tf.float64, shape=[None, 25], name='inp_feed')
    # tf_feasible_action = tf_constrained_project_nearest_L2(
    #     tf_inp_feed, c, cmin, cmax)
    # with tf.Session() as sess:
    #     feasible_action = timeit(lambda: sess.run(tf_feasible_action, feed_dict={
    #         tf_inp_feed: [test_input]
    #     }), repeat=100)
    #     print('feasible action:\n', feasible_action * 32)

    tf_y = tf.Variable(initial_value=np.array([ti_3]), name='y')
    tf_z = tf_constrained_project_nearest_L2(tf_y, c, cmin, cmax, name='z')
    tf_target = tf.constant(np.array([np.array([1 / 32] * 25)]))
    _loss = tf.reduce_mean(tf.square(tf_z - tf_target))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    _grads = tf.gradients(_loss, [tf_y])
    _grads = [tf.clip_by_value(g, -1, 1) for g in _grads]
    sgd_step = optimizer.apply_gradients(zip(_grads, [tf_y]))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(tf_y)
        print('y', y * 32)
        for step in range(1000):
            _, grad, y, z, loss = sess.run([sgd_step, _grads, tf_y, tf_z, _loss])
            print('step, loss, y, z:', step, loss, y * 32, z * 32)


if __name__ == '__main__':
    test_optnet()
