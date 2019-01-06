import os

import cplex
import numpy as np
import tensorflow as tf

from RL.common.utils import py_func, tf_scale


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
    if os.getenv('RL_CP_OPTNET_CPLEX_QPMETHOD', None) is not None:
        alg = int(os.getenv('RL_CP_OPTNET_CPLEX_QPMETHOD'))
        prob.parameters.qpmethod.set(alg)

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
    z, lambda_, alpha, beta = cplex_batch_CP_with_duals(y, c, cmin, cmax, k)
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
    A_inv = np.linalg.inv(A)
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
            with tf.variable_scope('input_scaling'):
                tf_cmin = tf.reshape(tf.constant(
                    cmin, dtype=tf.float32), [-1, k])
                tf_cmax = tf.reshape(tf.constant(
                    cmax, dtype=tf.float32), [-1, k])
                max_y = tf.maximum(tf.reduce_max(
                    y, axis=-1, keepdims=True), tf_cmax)
                min_y = tf.minimum(tf.reduce_min(
                    y, axis=-1, keepdims=True), tf_cmin)
                y_dash = tf_scale(y, min_y, max_y, tf_cmin,
                                  tf_cmax, "scale_to_cmin_cmax")
        else:
            y_dash = y
        ans = py_func(cplex_batch_cp, [y_dash, c, cmin, cmax, k],
                      tf.float64, name=scope, grad=tf_cplex_batch_CP_gradients)
        ans = tf.cast(ans, tf.float32)
        ans = tf.reshape(ans, [-1, k])
        return ans
