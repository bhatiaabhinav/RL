import os
import sys
import time

import cplex  # noqa: F401
import numpy as np
import tensorflow as tf

from RL.common.plot_renderer import PlotRenderer

optnet_layer = None
optnet_jacobians = None
tf_optnet_layer = None

use_custom_optnet = int(os.getenv('RL_USE_APPRX_OPTNET', "1"))
if use_custom_optnet:
    from RL.reco_rl.optnet_custom import custom_batch_cp, custom_batch_cp_jacobian, tf_custom_batch_cp
    optnet_layer = custom_batch_cp
    optnet_jacobians = custom_batch_cp_jacobian
    tf_optnet_layer = tf_custom_batch_cp
else:
    from RL.reco_rl.optnet_cplex import cplex_batch_cp, cplex_batch_CP_jacobian, tf_cplex_batch_CP
    optnet_layer = cplex_batch_cp
    optnet_jacobians = cplex_batch_CP_jacobian
    tf_optnet_layer = tf_cplex_batch_CP


def timeit(f, repeat=1000):
    t_start = time.time()
    for i in range(repeat):
        ans = f()
    t_end = time.time()
    fps = repeat / (t_end - t_start)
    print('fps', fps)
    return ans


def basic_forward_pass_and_grad_tests(test_input, c, cmin, cmax, k):
    feasible_action = optnet_layer(test_input, c, cmin, cmax, k)
    grads = optnet_jacobians(test_input, c, cmin, cmax, k)[0]
    np.set_printoptions(precision=2)
    print('feasible action:\n', feasible_action * 32)
    print('grads:\n', np.round(grads, 3))
    # if it is feasible action, doing another forward pass should not change it:
    feasible_new = optnet_layer(feasible_action, c, cmin, cmax, k)
    print('feasible action after another pass:\n', feasible_new * 32)
    # when the action is feasible, expect jacobian to be identity
    grads = optnet_jacobians(feasible_action, c, cmin, cmax, k)[0]
    print('gradients at a feasible action should be identity:\n', np.round(grads, 3))


def timing_test(test_input, c, cmin, cmax, k):
    tf_y = tf.Variable(initial_value=np.array(
        [test_input] * 64), dtype=tf.float32, name='y')
    tf_z = tf_optnet_layer(tf_y, c, cmin, cmax, k, scale_inputs=False)
    tf_target = np.array([1 / 32] * 3 + [2 / 32] * 5 + [0] * 4 + [4 / 32] +
                         [3 / 32, 1 / 32, 0 / 32, 2 / 32, 1 / 32] + [0] * 5 + [6 / 32] * 1 + [2 / 32])
    _loss = tf.reduce_mean(tf.square(tf_z - tf_target))
    _grads = tf.gradients(_loss, [tf_y])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        timeit(lambda: sess.run(_grads), repeat=1000)


def point_attraction_training_test(test_input, c, cmin, cmax, k):
    contrived_input = [-1] * 6 + [1] * 6 + [0.5] * 13  # noqa: F841
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
    tf_z = tf_optnet_layer(tf_y, c, cmin, cmax, k, name='z', scale_inputs=True)
    tf_target = tf.constant(np.array([z_target]), dtype=tf.float32)
    _loss = tf.reduce_mean(tf.square(tf_z - tf_target))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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


def test_optnet(test):
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
    test(test_input, c, cmin, cmax, k)


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else "timing"
    test_map = {
        "point_attraction": point_attraction_training_test,
        "basic": basic_forward_pass_and_grad_tests,
        "timing": timing_test
    }
    test_optnet(test_map.get(arg))
