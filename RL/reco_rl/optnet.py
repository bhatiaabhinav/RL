import os
import time

import cplex  # noqa: F401
import numpy as np
import scipy
import tensorflow as tf

from RL.common.plot_renderer import PlotRenderer
from RL.common.utils import py_func, tf_scale


def _cplex_batch_CP(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k, N):
    """
    solves QP 13 from document ddpg-cp-qp-optnet.pdf
    obj = min sum (z_s_i - y_s_i)^2
    subject to:
        forall s in 1..N: sum z_s_i = C_s   (lambda_s)
        forall s in 1..N: forall i in 1..k: z_s_i <= cmax_s_i         (alpha_s_i)
        forall s in 1..N: forall i in 1..k: z_s_i >= cmin_s_i         (beta_s_i)
    where lambda_s, alpha_s_i, beta_s_i are lagrange multipliers

    Arguments:
        y -- inputs tensor. [N, k]. float64
        c -- sum constraints tensor. [N] float64
        cmin -- min constraints tensor. [N, k] float64
        cmax -- max constraints tensor. [N, k] float64
        N - batch size
        k - dimensionality of the problem.
    Returns:
        z, lambda, alpha, beta, cplex_problem. All float64
    """
    z_names_flat = []
    for s in range(N):
        for i in range(k):
            z_names_flat.append('z_{0}_{1}'.format(s, i))

    prob = cplex.Cplex()
    if os.getenv('RL_CP_OPTNET_CPLEX_THREADS', None) is not None:
        max_threads = int(os.getenv('RL_CP_OPTNET_CPLEX_THREADS'))
        prob.parameters.threads.set(max_threads)

    # objective:
    y_flat = y.flatten()
    assert len(y_flat) == N * k
    prob.objective.set_sense(prob.objective.sense.minimize)
    obj = list(-2 * y_flat)
    prob.variables.add(obj=obj, names=z_names_flat)
    # since Q=2I, i.e. a seperable quadratic objective, we can just give coeffs of z_i^2
    qmat = 2 * np.ones(N * k)
    prob.objective.set_quadratic(qmat)

    row_names = []
    lhss = []
    senses = []
    rhss = []

    # lambda_:
    for s in range(N):
        row_names.append('lambda_{0}'.format(s))
        lhss.append([z_names_flat[s * k:(s + 1) * k], [1] * k])
        senses.append('E')
        rhss.append(c[s])

    # alpha_s_i:
    for s in range(N):
        for i in range(k):
            row_names.append('alpha_{0}_{1}'.format(s, i))
            lhss.append([['z_{0}_{1}'.format(s, i)], [1]])
            senses.append('L')
            rhss.append(cmax[s, i])

    # beta_s_i:
    for s in range(N):
        for i in range(k):
            row_names.append('beta_{0}_{1}'.format(s, i))
            lhss.append([['z_{0}_{1}'.format(s, i)], [-1]])
            senses.append('L')
            rhss.append(-cmin[s, i])

    assert len(row_names) == len(lhss) == len(senses) == len(rhss) == 2 * N * k + N, "There should be 2Nk+N={0} constraints. But found {1}".format(
        2 * N * k + N, len(row_names))

    prob.linear_constraints.add(
        lin_expr=lhss, senses=senses, rhs=rhss, names=row_names)

    prob.set_results_stream(None)
    prob.set_log_stream(None)

    prob.solve()

    z = np.array(prob.solution.get_values()).reshape([N, k])
    dual_vars = np.array(prob.solution.get_dual_values())
    assert len(dual_vars) == 2 * N * k + \
        N, "Number of dual varaibles should have been same as no. of constaints"
    lambda_ = dual_vars[0:N]
    alpha = dual_vars[N:N + N * k].reshape([N, k])
    beta = dual_vars[N + N * k:].reshape([N, k])
    return z, lambda_, alpha, beta, prob


def _preprocess_input_for_cplex_batch_CP(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k):
    y = np.asarray(y, float)
    c = np.asarray(c, float)
    cmin = np.asarray(cmin, float)
    cmax = np.asarray(cmax, float)
    y = np.reshape(y, [-1, k])
    c = np.reshape(c, [-1])
    cmin = np.reshape(cmin, [-1, k])
    cmax = np.reshape(cmax, [-1, k])
    N = y.shape[0]
    if c.shape[0] == 1:
        c = np.tile(c, N)
    if cmin.shape[0] == 1:
        cmin = np.tile(cmin, [N, 1])
    if cmax.shape[0] == 1:
        cmax = np.tile(cmax, [N, 1])
    assert y.shape[0] == c.shape[0] == cmin.shape[0] == cmax.shape[0], "inconsistent input. The shapes are {0},{1},{2},{3}".format(
        y.shape, c.shape, cmin.shape, cmax.shape)
    return y, c, cmin, cmax, k, N


def cplex_batch_CP_with_duals(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k):
    """
    solves QP 13 from document ddpg-cp-qp-optnet.pdf
    obj = min sum (z_s_i - y_s_i)^2
    subject to:
        forall s in 1..N: sum z_s_i = C_s   (lambda_s)
        forall s in 1..N: forall i in 1..k: z_s_i <= cmax_s_i   (alpha_s_i)
    where lambda_s, alpha_s_i are lagrange multipliers
    Batch size N will be automatically inferred from the inputs.
    Arguments:
        y -- inputs tensor. [N, k]
        c -- sum constraints tensor. [N]
        cmax -- max constraints tensor. [N, k]
        k - dimensionality of the problem.
    Returns:
        z, lambda, alpha. All float64
    """
    y, c, cmin, cmax, k, N = _preprocess_input_for_cplex_batch_CP(
        y, c, cmin, cmax, k)
    z, lambda_, alpha, beta, prob = _cplex_batch_CP(y, c, cmin, cmax, k, N)
    # prob.write('batchqp.lp')
    return z, lambda_, alpha, beta


def cplex_batch_cp(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k):
    """Same as `cplex_batch_cp` except that it does not return the dual variables"""
    z, lambda_, alpha, beta = cplex_batch_CP_with_duals(y, c, cmin, cmax, k)
    return z


def _batch_KKT_diff_equations_coeffs_matrix(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k):
    """
    Returns the matrix A per sample.
    Refer to equations 6 in the notes.
    Thus returns A of shape N,3k+1,3k+1
    """
    y, c, cmin, cmax, k, N = _preprocess_input_for_cplex_batch_CP(
        y, c, cmin, cmax, k)
    print("Doing forward pass with batch of " + str(N))
    start_t = time.time()
    z, lambda_, alpha, beta = cplex_batch_CP_with_duals(y, c, cmin, cmax, k)
    print("Forward pass done in {0}".format(time.time() - start_t))
    A = np.zeros([N, 3 * k + 1, 3 * k + 1])

    for s in range(N):
        for m in range(1, k + 1):
            A[s, 0, 3 * m] = 1
            A[s, 3 * m - 2, 0] = 1
            A[s, 3 * m - 2, 3 * m - 2] = 1
            A[s, 3 * m - 2, 3 * m - 1] = -1
            A[s, 3 * m - 2, 3 * m] = 2
            A[s, 3 * m - 1, 3 * m - 2] = z[s, m - 1] - cmax[s, m - 1]
            A[s, 3 * m - 1, 3 * m] = alpha[s, m - 1]
            A[s, 3 * m, 3 * m - 1] = -z[s, m - 1] + cmin[s, m - 1]
            A[s, 3 * m, 3 * m] = -beta[s, m - 1]

    return A


def cplex_batch_CP_jacobian(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k):
    '''This function can be moved completely to tensorflow'''
    y, c, cmin, cmax, k, N = _preprocess_input_for_cplex_batch_CP(
        y, c, cmin, cmax, k)
    A = _batch_KKT_diff_equations_coeffs_matrix(y, c, cmin, cmax, k)
    print("inverting matrix")
    start_t = time.time()
    A_inv = np.linalg.inv(A)
    # A_inv = np.zeros(A.shape)
    # for idx in range(N):
    #     A_sparse = scipy.sparse.csc_matrix(A[idx])
    #     A_sparse_inv = scipy.sparse.linalg.inv(A_sparse)
    #     A_inv[idx] = A_sparse_inv.todense()
    print("inverted in {0}".format(time.time() - start_t))
    J_y = np.zeros([N, 3 * k + 1, k])  # jacobian of dims N x z x y
    for j in range(k):
        # for derivatives w.r.t y_j. A_inv x B^y_j
        J_y[:, :, j] = 2 * A_inv[:, :, 3 * j + 1]  # N x (3k+1)
    indices_corresponding_to_zs = 3 + 3 * np.arange(k)
    J_y = J_y[:, indices_corresponding_to_zs, :]  # [N, k, k]

    J_c = np.zeros([N, 3 * k + 1, 1])  # jacobian of dims N x z x 1
    # for derivatives w.r.t c. A_inv x B^C
    J_c[:, :, 0] = A_inv[:, :, 0]  # N x (3k+1)
    J_c = J_c[:, indices_corresponding_to_zs, :]  # [N, k, 1]
    return J_y, J_c


def tf_cplex_batch_CP_gradients(op, grads_wrt_z):
    # print("gradient graph being set up")
    grads_wrt_z = tf.expand_dims(grads_wrt_z, 1)  # [N, 1, k]
    y = op.inputs[0]  # [N, k]
    c = op.inputs[1]  # [N]
    cmin = op.inputs[2]  # [N, k]
    cmax = op.inputs[3]  # [N, k]
    y_shape = y.get_shape().as_list()
    c_shape = c.get_shape().as_list()
    cmin_shape = cmin.get_shape().as_list()
    cmax_shape = cmax.get_shape().as_list()
    k = y_shape[1] if len(y_shape) == 2 else y_shape[0]

    J_z_wrt_y, J_z_wrt_c = tf.py_func(cplex_batch_CP_jacobian, [y, c, cmin, cmax, k], [
                                      tf.float64, tf.float64], name='cplex_batch_CP_jacobians')
    grads_wrt_y = tf.reshape(
        tf.matmul(grads_wrt_z, J_z_wrt_y), [-1, k])  # will give [N, k]
    if len(y_shape) == 1:
        grads_wrt_y = grads_wrt_y[0]
    grads_wrt_c = tf.reshape(
        tf.matmul(grads_wrt_z, J_z_wrt_c), [-1])  # will give [N]
    if len(c_shape) == 0:
        grads_wrt_c = grads_wrt_c[0]
    grads_wrt_y = tf.cast(grads_wrt_y, tf.float32)
    grads_wrt_c = tf.cast(grads_wrt_c, tf.float32)
    grads_wrt_cmin = tf.constant(0, dtype=tf.float32, shape=cmin_shape)
    grads_wrt_cmax = tf.constant(0, dtype=tf.float32, shape=cmax_shape)
    grads_wrt_k = tf.constant(0, dtype=tf.float32)
    grads_wrt_y = tf.clip_by_value(
        grads_wrt_y, -10, 10)  # just to ensure stability
    grads_wrt_c = tf.clip_by_value(
        grads_wrt_c, -10, 10)  # just to ensure stability
    return grads_wrt_y, grads_wrt_c, grads_wrt_cmin, grads_wrt_cmax, grads_wrt_k


def tf_cplex_batch_CP(y, c, cmin, cmax, k, scale_inputs=True, name='cplex_batch_CP'):
    with tf.name_scope(name, "cplex_batch_CP", [y, c, cmin, cmax, k]) as scope:
        if scale_inputs:
            y_dash = 2 * tf.nn.tanh(y, name='tanh')
            # y_dash = tf.sin(y)
            # p = 4
            # y_dash = (4 / p) * (tf.abs(tf.mod(y - 1, p) - p / 2) - p / 4)
            # y_dash = y
            y_dash = tf_scale(y_dash, -1, 1, cmin, cmax, "scale_to_cmin_cmax")
        else:
            y_dash = y
        ans = py_func(cplex_batch_cp, [y_dash, c, cmin, cmax, k],
                      tf.float64, name=scope, grad=tf_cplex_batch_CP_gradients)
        ans = tf.cast(ans, tf.float32)
        ans = tf.reshape(ans, [-1, k])
        return ans


def timeit(f, repeat=1000):
    t_start = time.time()
    for i in range(repeat):
        ans = f()
    t_end = time.time()
    fps = repeat / (t_end - t_start)
    print('fps', fps)
    return ans


def basic_forward_pass_and_grad_tests(test_input, c, cmin, cmax, k):
    feasible_action = cplex_batch_cp(test_input, c, cmin, cmax, k)
    grads = cplex_batch_CP_jacobian(test_input, c, cmin, cmax, k)[0]
    np.set_printoptions(precision=2)
    print('feasible action:\n', feasible_action * 32)
    print('grads:\n', np.round(grads, 3))
    # if it is feasible action, doing another forward pass should not change it:
    feasible_new = cplex_batch_cp(feasible_action, c, cmin, cmax, k)
    print('feasible action after another pass:\n', feasible_new * 32)
    # when the action is feasible, expect jacobian to be identity
    grads = cplex_batch_CP_jacobian(feasible_action, c, cmin, cmax, k)[0]
    print('gradients at a feasible action should be identity:\n', np.round(grads, 3))


def timing_test(test_input, c, cmin, cmax, k):
    tf_inp_feed = tf.placeholder(tf.float32, shape=[None, 25], name='inp_feed')
    tf_feasible_action = tf_cplex_batch_CP(tf_inp_feed, c, cmin, cmax, k)
    with tf.Session() as sess:
        feasible_action = timeit(lambda: sess.run(tf_feasible_action, feed_dict={
            tf_inp_feed: [test_input] * 16
        }), repeat=1000)
    print('feasible action:\n', feasible_action * 32)


def point_attraction_training_test(test_input, c, cmin, cmax, k):
    contrived_input = [-1] * 6 + [1] * 6 + [0.5] * 13
    zero_input = [0] * 25  # noqa: F841
    good_input = [2 / 32] * 6 + [4 / 32] * 6 + [3 / 32] * 6 + [1 / 32] * 7  # noqa: F841
    mid_input = [0.1] * 25  # noqa: F841
    z_target = np.array([1 / 32] * 3 + [2 / 32] * 5 + [0] * 4 + [4 / 32] +
                        [3 / 32, 1 / 32, 0 / 32, 2 / 32, 1 / 32] + [0] * 5 + [6 / 32] * 1 + [2 / 32])
    assert np.sum(z_target) == 1
    assert len(z_target) == k
    y_init = contrived_input
    tf_y = tf.Variable(initial_value=np.array(
        [y_init]), dtype=tf.float32, name='y')
    tf_z = tf_cplex_batch_CP(tf_y, c, cmin, cmax, k, name='z')
    tf_target = tf.constant(np.array([z_target]), dtype=tf.float32)
    _loss = tf.reduce_mean(tf.square(tf_z - tf_target))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    _grads = tf.gradients(_loss, [tf_y])
    sgd_step = optimizer.apply_gradients(zip(_grads, [tf_y]))
    grads_max, grads_min, grads_avg = [], [], []
    loss_data = []
    grads_renderer = PlotRenderer(
        title='Gradients', xlabel='step', ylabel='value', window_caption='Optnet Test', save_path='Optnet Test/gradients.png', auto_save=True)
    loss_renderer = PlotRenderer(
        title='Loss', xlabel='step', ylabel='loss', window_caption='Optnet Test', save_path='Optnet Test/loss.png', auto_save=True)
    grads_renderer.plot([], grads_max, [], grads_avg, [], grads_min)
    loss_renderer.plot([], loss_data)
    z_renderer = PlotRenderer(title='z actual vs target', xlabel='base_id',
                              ylabel='alloc', window_caption='Optnet Test', save_path='Optnet Test/z.png', auto_save=True)
    z_renderer.plot(list(range(k)), z_target * 32,
                    'b-', list(range(k)), [0] * k, 'g')
    y_renderer = PlotRenderer(
        title='y', xlabel='base_id', ylabel='value', window_caption='Optnet Test', save_path='Optnet Test/y.png', auto_save=True)
    y_renderer.plot(list(range(k)), y_init)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        y = sess.run(tf_y)
        for step in range(5000):
            _, grad, y, z, loss = sess.run(
                [sgd_step, _grads, tf_y, tf_z, _loss])
            grads_max.append(np.max(grad))
            grads_min.append(np.min(grad))
            grads_avg.append(np.average(grad))
            loss_data.append(np.log(loss))
            if step % 10 == 0:
                x = list(range(len(loss_data)))
                grads_renderer.update_and_render(
                    [[x, grads_max], [x, grads_avg], [x, grads_min]])
                loss_renderer.update_and_render([[x, loss_data]])
                z_renderer.update_and_render(
                    [[list(range(k)), z_target * 32], [list(range(k)), z * 32]])
                y_renderer.update_and_render([[list(range(k)), y * 32]])
            # print('step, loss, y, z:', step, loss, y * 32, z * 32)
    time.sleep(10)
    grads_renderer.close()
    loss_renderer.close()


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
    k = 25
    test_input = np.asarray(
        [5.5, 4, 3, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0]) / 32
    # basic_forward_pass_and_grad_tests(test_input, c, cmin, cmax, k)
    # timing_test(test_input, c, cmin, cmax, k)
    point_attraction_training_test(test_input, c, cmin, cmax, k)


if __name__ == '__main__':
    test_optnet()
