import os

import numpy as np
import tensorflow as tf

from RL.common.utils import py_func, tf_scale

use_extra_grads = int(os.getenv('RL_APPRX_OPTNET_USE_LEAKY_GRADS', "0"))


def _preprocess_input_for_custom_batch_cp(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k):
    y = np.asarray(y, np.float32)
    c = np.asarray(c, np.float32)
    cmin = np.asarray(cmin, np.float32)
    cmax = np.asarray(cmax, np.float32)
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


def _custom_batch_cp_fast(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k, N, want_jacobian=False):
    z = np.zeros([N, k], dtype=np.float32)
    if want_jacobian:
        J = np.zeros([N, k, k], dtype=np.float32)
    for n in range(N):
        _c = c[n]  # a scalar
        _k = k  # an integer
        unallocated_indices_mask = np.ones(k, dtype=np.float32)
        lower_bound_clamping_done = False
        violations_exist = True
        while violations_exist:
            z[n] = (1 - unallocated_indices_mask) * z[n] + unallocated_indices_mask * \
                (y[n] + (_c - np.sum(y[n] * unallocated_indices_mask)) / _k)
            if not lower_bound_clamping_done:
                violating_indices_mask_new = (
                    z[n] < cmin[n]).astype(np.float32)
                z_clamped_new = cmin[n]
                if np.sum(violating_indices_mask_new) == 0:
                    lower_bound_clamping_done = True
            if lower_bound_clamping_done:
                violating_indices_mask_new = (
                    z[n] > cmax[n]).astype(np.float32)
                z_clamped_new = cmax[n]
                if np.sum(violating_indices_mask_new) == 0:
                    violations_exist = False
            if violations_exist:
                # clamp the new violating zs
                z[n] = (1 - violating_indices_mask_new) * z[n] + \
                    violating_indices_mask_new * z_clamped_new
                _k = _k - np.sum(violating_indices_mask_new, dtype=np.float32)
                _c = _c - \
                    np.sum(z[n] * violating_indices_mask_new, dtype=np.float32)
                assert _c >= 0, "not enough sum left"
                unallocated_indices_mask = unallocated_indices_mask - \
                    violating_indices_mask_new.astype(np.float32)
        if want_jacobian:
            if use_extra_grads:
                grads_z = np.identity(k, dtype=np.float32) - \
                    (1 / k) * np.ones([k, k])
                J[n] = grads_z
            else:
                grads_z = np.identity(k, dtype=np.float32) - \
                    (1 / _k) * np.ones([k, k])
                grads_mask = np.outer(
                    unallocated_indices_mask, unallocated_indices_mask)
                J[n] = grads_z * grads_mask
    if want_jacobian:
        return z, J
    else:
        return z


def _custom_batch_cp_fastest(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k, N, want_jacobian=False):
    _c = np.reshape(c.copy(), [N, 1])
    _k = k * np.ones([N, 1], dtype=np.float32)
    z = np.zeros([N, k], dtype=np.float32)
    unallocated_indices_mask = np.ones([N, k], dtype=np.float32)
    violations_exist = 1
    while violations_exist:
        z = (1 - unallocated_indices_mask) * z + unallocated_indices_mask * (y +
                                                                             (_c - np.sum(y * unallocated_indices_mask, axis=-1, keepdims=True)) / _k)
        lower_bound_clamping_done = np.all(
            z >= cmin, axis=-1, keepdims=True).astype(np.float32)
        violating_indices_mask_new = (1 - lower_bound_clamping_done) * (
            z < cmin).astype(np.float32) + lower_bound_clamping_done * (z > cmax).astype(np.float32)
        z_clamped_new = (1 - lower_bound_clamping_done) * \
            cmin + lower_bound_clamping_done * cmax
        violations_exist = 1 - \
            int(np.all(lower_bound_clamping_done) and
                not np.any(violating_indices_mask_new))
        if violations_exist:
            # clamp the new violating zs
            z = (1 - violating_indices_mask_new) * z + \
                violating_indices_mask_new * z_clamped_new
            _k = _k - np.sum(violating_indices_mask_new,
                             axis=-1, keepdims=True, dtype=np.float32)
            _c = _c - np.sum(z * violating_indices_mask_new,
                             axis=-1, keepdims=True, dtype=np.float32)
            assert np.all(_c >= 0), "not enough sum left"
            unallocated_indices_mask = unallocated_indices_mask - \
                violating_indices_mask_new.astype(np.float32)
    if want_jacobian:
        if use_extra_grads:
            grads_z = np.array([np.identity(k, dtype=np.float32)] * N,
                               dtype=np.float32) - (1 / k) * np.ones([N, k, k], dtype=np.float32)
            J = grads_z
        else:
            grads_z = np.array([np.identity(k, dtype=np.float32)] * N, dtype=np.float32) - (
                1 / np.reshape(_k, [N, 1, 1])) * np.ones([N, k, k], dtype=np.float32)
            grads_mask = np.matmul(np.expand_dims(
                unallocated_indices_mask, axis=-1), np.expand_dims(unallocated_indices_mask, axis=-2))
            J = grads_z * grads_mask
        return z, J
    else:
        return z


def custom_batch_cp(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k):
    # print("Forward pass")
    y, c, cmin, cmax, k, N = _preprocess_input_for_custom_batch_cp(
        y, c, cmin, cmax, k)
    z = _custom_batch_cp_fastest(y, c, cmin, cmax, k, N)
    return z


def custom_batch_cp_jacobian(y: np.ndarray, c: np.ndarray, cmin: np.ndarray, cmax: np.ndarray, k):
    # print("Backward pass")
    y, c, cmin, cmax, k, N = _preprocess_input_for_custom_batch_cp(
        y, c, cmin, cmax, k)
    z, J = _custom_batch_cp_fastest(y, c, cmin, cmax, k, N, want_jacobian=True)
    # not implementing jacobian for c right now
    return J, np.zeros([N, k], dtype=np.float32)


def tf_custom_batch_cp_gradients(op, grads_wrt_z):
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
    J_z_wrt_y, J_z_wrt_c = tf.py_func(custom_batch_cp_jacobian, [y, c, cmin, cmax, k], [
                                      tf.float32, tf.float32], name='custom_CP_jacobian')
    grads_wrt_y = tf.reshape(
        tf.matmul(grads_wrt_z, J_z_wrt_y), [-1, k])  # will give [N, k]
    if len(y_shape) == 1:
        grads_wrt_y = grads_wrt_y[0]
    grads_wrt_c = tf.constant(0, dtype=tf.float32, shape=c_shape)
    grads_wrt_cmin = tf.constant(0, dtype=tf.float32, shape=cmin_shape)
    grads_wrt_cmax = tf.constant(0, dtype=tf.float32, shape=cmax_shape)
    grads_wrt_k = tf.constant(0, dtype=tf.float32)
    return grads_wrt_y, grads_wrt_c, grads_wrt_cmin, grads_wrt_cmax, grads_wrt_k


def tf_custom_batch_cp(y, c, cmin, cmax, k, scale_inputs=True, name='custom_batch_CP'):
    with tf.name_scope(name, "custom_batch_CP", [y, c, cmin, cmax, k]) as scope:
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
        ans = py_func(custom_batch_cp, [y_dash, c, cmin, cmax, k],
                      tf.float32, name=scope, grad=tf_custom_batch_cp_gradients)
        ans = tf.reshape(ans, [-1, k])
        return ans
