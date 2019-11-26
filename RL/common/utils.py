import os
import os.path
import random
from functools import reduce
from operator import mul

import joblib
import numpy as np
import pyglet
import tensorflow as tf
import tensorflow.contrib as tc
import gym
from keras.initializers import Orthogonal
from keras.layers import Activation, BatchNormalization, Dense
from keras.models import Sequential
from keras.optimizers import Adam

import RL


class ImagePygletWingow(pyglet.window.Window):
    def __init__(self, width=None, height=None, caption=None, resizable=True, style=None, fullscreen=False, visible=True, vsync=True, display=None, screen=None, config=None, context=None, mode=None):
        super().__init__(width=width, height=height, caption=caption, resizable=resizable, style=style, fullscreen=fullscreen, visible=visible, vsync=vsync, display=display, screen=screen, config=config, context=context, mode=mode)
        self.image = None
        self.needs_draw = True
        if height is None or width is None:
            self.auto_resize = True

    def on_draw(self):
        if self.image is not None:
            self.image.blit(0, 0, width=self.width, height=self.height)

    def set_image(self, arr, dispatch_events=False):
        assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
        self.image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
        self.needs_draw = True
        if self.auto_resize:
            self.set_size(arr.shape[1], arr.shape[0])
            self.auto_resize = False
        if dispatch_events:
            self.switch_to()
            self.dispatch_events()
            if self.needs_draw:
                self.needs_draw = False
                # self.clear()
                self.dispatch_event('on_draw')
                self.flip()

    def imshow(self, arr):
        self.set_image(arr, True)

    def set_text_image(self, text, dispatch_events=False):
        from PIL import Image, ImageDraw, ImageFont
        # Availability is platform dependent
        size = 20
        font = 'cour'
        # Create font
        pil_font = ImageFont.truetype(font + ".ttf", size=size)
        text_width, text_height = pil_font.getsize(text)

        # create a blank canvas with extra space between lines
        canvas = Image.new('RGB', [self.width, self.height], (255, 255, 255))

        # draw the text onto the canvas
        draw = ImageDraw.Draw(canvas)
        offset = (10, 10)
        white = "#000000"
        draw.text(offset, text, font=pil_font, fill=white)
        # Convert the canvas into an array with values in [0, 1]
        img = (255 - np.asarray(canvas)).astype(np.uint8)
        self.set_image(img, dispatch_events=dispatch_events)

    def textimshow(self, text):
        self.set_text_image(text, True)


class SimpleImageViewer(object):
    def __init__(self, display=None, width=None, height=None, caption='SimpleImageViewer', resizable=True, vsync=False):
        self.window = None
        self.isopen = False
        self.display = display
        self.width = width
        self.height = height
        self.resizable = resizable
        self.vsync = vsync
        self.caption = caption
        self._failed = False
        self._image = None

    def _initialize(self, height=None, width=None):
        if self.height is None:
            self.height = height
        if self.width is None:
            self.width = width
        try:
            self.window = pyglet.window.Window(
                width=self.width, height=self.height, display=self.display, vsync=self.vsync, resizable=self.resizable, caption=self.caption)
        except Exception as e:
            self.window = None
            self._failed = True
            RL.logger.warn("Could not create window: {0}".format(e))

        if not self._failed:
            self.isopen = True

            @self.window.event
            def on_resize(width, height):
                self.width = width
                self.height = height

            @self.window.event
            def on_close():
                self.isopen = False

            @self.window.event
            def on_draw():
                self.window.clear()
                if self._image is not None:
                    self._image.blit(
                        0, 0, width=self.window.width, height=self.window.height)

    def imshow(self, arr):
        if not self._failed:
            self.set_image(arr)
            # self.window.clear()
            self.window.switch_to()
            self.window.dispatch_events()
            self.window.dispatch_event('on_draw')
            self.window.flip()

    def set_image(self, arr):
        if self.window is None and not self._failed:
            self._initialize(arr.shape[0], arr.shape[1])
        if not self._failed:
            assert len(arr.shape) == 3, "You passed in an image with the wrong number shape"
            self._image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def dispatch_events(self):
        if self.isopen:
            self.window.dispatch_events()

    def __del__(self):
        self.close()


class GridImageViewer(object):
    def __init__(self, n_vertical, n_horizontal, display=None, width=None, height=None, caption='GridImageViewer', resizable=True, vsync=False):
        self.window = None
        self.isopen = False
        self.display = display
        self.width = width
        self.height = height
        self.resizable = resizable
        self.vsync = vsync
        self.caption = caption
        self._failed = False
        self.n_vertical = n_vertical
        self.n_horizontal = n_horizontal
        self.padding = 10
        self._images = []

    @property
    def img_height(self):
        return max(int((self.height - self.padding * (self.n_vertical + 1)) / self.n_vertical), 10)

    @property
    def img_width(self):
        return max(int((self.width - self.padding * (self.n_horizontal + 1)) / self.n_horizontal), 10)

    def imshow(self, arr_list):
        if self.window is None and not self._failed:
            height = int(np.mean([arr.shape[0] for arr in arr_list]) * self.n_vertical + self.padding * (self.n_vertical + 1))
            width = int(np.mean([arr.shape[1] for arr in arr_list]) * self.n_horizontal + self.padding * (self.n_horizontal + 1))
            if self.height is None:
                self.height = height
            if self.width is None:
                self.width = width
            try:
                self.window = pyglet.window.Window(
                    width=self.width, height=self.height, display=self.display, vsync=self.vsync, resizable=self.resizable, caption=self.caption)
            except Exception as e:
                self.window = None
                self._failed = True
                RL.logger.warn("Could not create window: {0}".format(e))

            if not self._failed:
                self.isopen = True

                @self.window.event
                def on_resize(width, height):
                    self.width = width
                    self.height = height

                @self.window.event
                def on_close():
                    self.isopen = False

                @self.window.event
                def on_draw():
                    pyglet.gl.glClearColor(1, 1, 1, 1)
                    self.window.clear()
                    top = self.padding
                    left = self.padding
                    count = 0
                    for image in self._images:
                        image.blit(left, self.height - top - self.img_height,
                                   width=self.img_width, height=self.img_height)
                        left += self.img_width + self.padding
                        count += 1
                        if count % self.n_horizontal == 0:
                            top += (self.img_height + self.padding)
                            left = self.padding
                self._on_draw = on_draw

        if not self._failed:
            self._images.clear()
            for arr in arr_list:
                assert arr.shape[2] == 3, "You passed in an image with the wrong number shape"
                image = pyglet.image.ImageData(
                    arr.shape[1], arr.shape[0], 'RGB', arr.tobytes(), pitch=arr.shape[1] * -3)
                self._images.append(image)
            self.window.switch_to()
            self._on_draw()
            self.window.flip()
            self.window.dispatch_events()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def dispatch_events(self):
        if self.isopen:
            self.window.dispatch_events()

    def __del__(self):
        self.close()


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)


def my_video_schedule(episode_id, total_episodes, video_interval):
    from gym.wrappers.monitor import capped_cubic_video_schedule
    if video_interval is not None and video_interval <= 0:
        return False
    if episode_id == total_episodes - 1:
        return True
    if video_interval is None:
        return capped_cubic_video_schedule(episode_id)
    return episode_id % video_interval == 0


def kdel(i, j):
    '''The Kronecker delta function. 1 when i=j. 0 otherwise'''

    return 1 if i == j else 0


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    """Defines a custom py_func which takes also a grad op as argument"""
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


class RunningStats:

    def __init__(self, shape, epsilon=1e-2):
        self.n = 0
        self.old_m = np.zeros(shape=shape)
        self.new_m = np.zeros(shape=shape)
        self.old_s = np.zeros(shape=shape)
        self.new_s = np.zeros(shape=shape)
        self.shape = shape
        self.epsilon = epsilon
        self.mean = np.zeros(shape=shape)
        self.variance = self.epsilon * np.ones(shape=shape)
        self.std = np.sqrt(self.variance)

    def update(self, x):
        x = np.asarray(x)
        if len(x.shape) == len(self.shape):
            x_batch = np.expand_dims(x, 0)
        elif len(x.shape) == len(self.shape) + 1:
            x_batch = x
        else:
            raise ValueError("bad shape of input")

        for i in range(len(x_batch)):
            x = x_batch[i]
            self.n += 1

            if self.n == 1:
                self.old_m = self.new_m = x
                self.old_s = np.zeros(shape=self.shape)
            else:
                self.new_m = self.old_m + (x - self.old_m) / self.n
                self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

                self.old_m = self.new_m
                self.old_s = self.new_s

            self.mean = self.new_m if self.n else np.zeros(shape=self.shape)
            variance = self.new_s / \
                (self.n - 1) if self.n > 1 else np.zeros(shape=self.shape)
            self.variance = np.maximum(variance, self.epsilon)
            self.std = np.sqrt(self.variance)

    def normalize(self, x):
        x = np.asarray(x)
        return (x - self.mean) / self.std

    def normalize_by_std(self, x):
        x = np.asarray(x)
        return x / self.std


class TfRunningStats:
    def __init__(self, shape, name, epsilon=1e-2, reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            self._n = tf.get_variable(
                "n", trainable=False, initializer=tf.constant(0.0))
            self._old_m = tf.get_variable(
                "old_m", trainable=False, initializer=tf.zeros(shape))
            self._new_m = tf.get_variable(
                "new_m", trainable=False, initializer=tf.zeros(shape))
            self._old_s = tf.get_variable(
                "old_s", trainable=False, initializer=tf.zeros(shape))
            self._new_s = tf.get_variable(
                "new_s", trainable=False, initializer=tf.zeros(shape))
            self.shape = shape
            self.epsilon = epsilon
            self.mean = tf.get_variable(
                "mean", trainable=False, initializer=tf.zeros(shape))
            self.variance = tf.get_variable(
                "variance", trainable=False, initializer=self.epsilon * tf.ones(shape))
            self.std = tf.get_variable(
                "std", trainable=False, initializer=np.sqrt(self.epsilon) * tf.ones(shape))

    def update(self, x, name="update_stats"):
        with tf.variable_scope(name):
            x = tf.cast(x, tf.float32)
            n = self._n + 1
            new_m = self._old_m + (x - self._old_m) / n
            old_m = new_m
            new_s = self._old_s + (x - self._old_m) * (x - new_m)
            old_s = tf.case(
                [(tf.equal(n, 1.0), lambda: tf.zeros(self.shape))],
                default=lambda: new_s
            )
            mean = new_m
            variance = tf.case(
                [(tf.greater(n, 1), lambda: new_s / (n - 1))],
                default=lambda: tf.zeros(shape=self.shape)
            )
            variance = tf.maximum(variance, self.epsilon)
            std = tf.sqrt(variance)

            return tf.group(
                tf.assign(self._n, n),
                tf.assign(self._old_m, old_m),
                tf.assign(self._new_m, new_m),
                tf.assign(self._old_s, old_s),
                tf.assign(self._new_s, new_s),
                tf.assign(self.mean, mean),
                tf.assign(self.variance, variance),
                tf.assign(self.std, std)
            )

    def normalize(self, x, name='normalize'):
        with tf.variable_scope(name):
            x = tf.cast(x, tf.float32)
            return (x - self.mean) / self.std

    def normalize_by_std(self, x, name='normalize_by_std'):
        with tf.variable_scope(name):
            x = tf.cast(x, tf.float32)
            return x / self.std


def normalize(a, epsilon=1e-6):
    a = np.clip(a, 0, 1)
    a = a + epsilon
    a = a / np.sum(a)
    return a


def scale(a, low, high, target_low, target_high):
    a_frac = (a - low) / (high - low)
    a = target_low + a_frac * (target_high - target_low)
    return a


def tf_scale(a, low, high, target_low, target_high, scope):
    with tf.variable_scope(scope):
        return scale(a, low, high, target_low, target_high)


def tf_clip(x, low=-1, high=1, name='clip'):
    with tf.variable_scope(name):
        return tf.minimum(tf.maximum(x, low), high)


def tf_safe_softmax(inputs, scope):
    with tf.variable_scope(scope):
        x = inputs - tf.reduce_max(inputs, axis=1, keepdims=True)
        exp = tf.exp(x)
        sigma = tf.reduce_sum(exp, axis=-1, keepdims=True, name='sum')
        return exp / sigma


def mutated_ers(alloc, max_mutations=2, mutation_rate=0.05):
    a = alloc.copy()
    for i in range(np.random.randint(1, max_mutations + 1)):
        src = np.random.randint(0, len(alloc))
        dst = np.random.randint(0, len(alloc))
        a[src] -= mutation_rate
        a[dst] += mutation_rate
    return normalize(a)


def mutated_gaussian(alloc, max_mutations=2, mutation_rate=0.05):
    a = alloc.copy()
    for i in range(np.random.randint(1, max_mutations + 1)):
        a = a + np.random.standard_normal(size=np.shape(a))
    a = np.clip(a, -1, 1)
    return a


def tf_log_transform(inputs, max_x, t, scope, is_input_normalized=True):
    with tf.variable_scope(scope):
        x = max_x * inputs
        return tf.log(1 + x / t) / tf.log(1 + max_x / t)


def tf_log_transform_adaptive(inputs, scope, max_inputs=1, uniform_gamma=False, gamma=None, shift=True, scale=True):
    with tf.variable_scope(scope):
        inputs_shape = inputs.shape.as_list()[1:]
        if gamma is None:
            if uniform_gamma:
                gamma = tf.Variable(1.0, name='gamma', dtype=tf.float32)
            else:
                gamma = tf.Variable(
                    np.ones(inputs_shape, dtype=np.float32), name='gamma')
            gamma = tf.square(gamma, name='gamma_squared')
            # gamma = tf.abs(gamma, name='gamma_abs')
            # gamma = tf.Print(gamma, [gamma])
        epsilon = 1e-3
        log_transform = tf.log(1 + gamma * inputs) / \
            (tf.log(1 + gamma * max_inputs) + epsilon)
        return log_transform


def color_to_grayscale(img):
    frame = np.dot(img.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
    frame = np.expand_dims(frame, 2)
    return np.concatenate([frame] * 3, axis=2)


def tf_normalize(inputs, mean, std, scope):
    with tf.variable_scope(scope):
        return (inputs - mean) / std


def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        # lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def conv(x, scope, nf, rf, stride, pad='VALID', act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[3].value
        w = tf.get_variable("w", [rf, rf, nin, nf],
                            initializer=ortho_init(init_scale))
        b = tf.get_variable(
            "bias", [nf], initializer=tf.constant_initializer(0.0))
        z = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=pad) + b
        h = act(z)
        return h


def conv_to_fc(x):
    nh = np.prod([v.value for v in x.get_shape()[1:]])
    x = tf.reshape(x, [-1, nh])
    return x


def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
        b = tf.get_variable(
            "bias", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = act(z)
        return h


def tf_conv_layers(inputs, inputs_shape, scope, use_ln=False, use_bn=False, training=False):
    with tf.variable_scope(scope):
        strides = [4, 2, 1]
        nfs = [32, 32, 64]
        rfs = [8, 4, 3]
        num_convs = 3 if inputs_shape[0] >= 60 else 2
        for i in range(3 - num_convs, 3):
            with tf.variable_scope('c{0}'.format(i + 1)):
                c = conv(
                    inputs, 'conv', nfs[i], rfs[i], strides[i], pad='VALID', act=lambda x: x)
                if use_ln:
                    # it is ok to use layer norm since we are using VALID padding. So corner neurons are OK.
                    c = tc.layers.layer_norm(
                        c, center=True, scale=True)
                elif use_bn:
                    c = tf.layers.batch_normalization(
                        c, training=training, name='batch_norm')
                c = tf.nn.relu(c, name='relu')
                inputs = c
        with tf.variable_scope('conv_to_fc'):
            flat = conv_to_fc(inputs)
    return flat


def tf_hidden_layers(inputs, scope, size, use_ln=False, use_bn=False, training=False):
    with tf.variable_scope(scope):
        for i in range(len(size)):
            with tf.variable_scope('h{0}'.format(i + 1)):
                h = tf.layers.dense(inputs, size[i], name='fc')
                if use_ln:
                    h = tc.layers.layer_norm(
                        h, center=True, scale=True)
                elif use_bn:
                    h = tf.layers.batch_normalization(
                        h, training=training, name='batch_norm')
                h = tf.nn.relu(h, name='relu')
                inputs = h
    return inputs


def tf_deep_net(inputs, inputs_shape, inputs_dtype, scope, hidden_size, init_scale=1.0, use_ln=False, use_bn=False, training=False, output_shape=None, bias=0):
    conv_needed = len(inputs_shape) > 1
    with tf.variable_scope(scope):
        if conv_needed:
            inp = tf.divide(tf.cast(inputs, tf.float32, name='cast_to_float'), 255., name='divide_by_255') \
                if inputs_dtype == 'uint8' else inputs
            flat = tf_conv_layers(inp, inputs_shape,
                                  'conv_layers', use_ln=use_ln, use_bn=use_bn, training=training)
        else:
            flat = inputs
        h = tf_hidden_layers(flat, 'hidden_layers',
                             hidden_size, use_ln=use_ln, use_bn=use_bn, training=training)
        if output_shape is not None:
            output_size = reduce(mul, output_shape, 1)
            # final_flat = tf.layers.dense(h, output_size, kernel_initializer=tf.random_uniform_initializer(
            #     minval=-init_scale, maxval=init_scale), name='output_flat')
            final_flat = tf.layers.dense(h, output_size, kernel_initializer=tf.orthogonal_initializer(
                gain=init_scale), bias_initializer=tf.constant_initializer(bias), name='output_flat')
            final = tf.reshape(
                final_flat, [-1] + list(output_shape), name='output')
        else:
            final = h
        return final


def conv_net(inputs, convs, activation_fn, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        prev_layer = tf.cast(inputs, tf.float32)
        for layer_id, (nfilters, kernel, stride) in enumerate(convs):
            h = tf.layers.conv2d(prev_layer, nfilters,
                                 kernel, stride, name="c{0}".format(layer_id))
            h = activation_fn(h)
            prev_layer = h
        prev_layer = tf.layers.flatten(prev_layer, name='flatten')
    return prev_layer


def dense_net(inputs, hidden_layers, activation_fn, output_size, output_activation, name, layer_norm=False, output_kernel_initializer=None, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        prev_layer = inputs
        for layer_id, layer in enumerate(hidden_layers):
            h = tf.layers.dense(prev_layer, layer,
                                activation=None, name='h{0}'.format(layer_id))
            if layer_norm:
                h = tc.layers.layer_norm(h, scale=True, center=True)
            h = activation_fn(h)
            prev_layer = h
        if output_size is not None:
            output_layer = tf.layers.dense(
                prev_layer, output_size, name='output', kernel_initializer=output_kernel_initializer)
            output_layer = output_activation(output_layer)
        else:
            output_layer = prev_layer
    return output_layer


def conv_dense_net(inputs, convs, hidden_layers, activation_fn, output_size, output_activation, name, layer_norm=False, output_kernel_initializer=None, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        conv_out = conv_net(inputs, convs, activation_fn, "conv_net")
        mlp_out = dense_net(conv_out, hidden_layers, activation_fn,
                            output_size, output_activation, "dense_net", layer_norm=layer_norm, output_kernel_initializer=output_kernel_initializer)
        return mlp_out


def auto_conv_dense_net(need_conv_net, inputs, convs, hidden_layers, activation_fn, output_size, output_activation, name, layer_norm=False, output_kernel_initializer=None, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        if need_conv_net:
            conv_out = conv_net(inputs, convs, activation_fn, "conv_net")
        else:
            conv_out = inputs
        mlp_out = dense_net(conv_out, hidden_layers, activation_fn,
                            output_size, output_activation, "dense_net", layer_norm=layer_norm, output_kernel_initializer=output_kernel_initializer)
        return mlp_out


def tf_l2_norm(vars, exclusions=['output', 'bias'], name='l2_losses'):
    with tf.variable_scope(name):
        loss = 0
        for v in vars:
            if not any(excl in v.name for excl in exclusions):
                loss += tf.nn.l2_loss(v)
        return loss


def tf_training_step(loss, vars, optimizer, l2_reg, clip_gradients, name, clip_gradients_by='norm'):
    with tf.variable_scope(name):
        if l2_reg:
            loss = loss + l2_reg * tf_l2_norm(vars)
        grads = tf.gradients(loss, vars)
        if clip_gradients:
            clip_gradients = float(clip_gradients)
            if clip_gradients_by == 'norm':
                grads = [tf.clip_by_norm(grad, clip_gradients) for grad in grads]
            elif clip_gradients_by == 'value':
                grads = [tf.clip_by_value(grad, -clip_gradients, clip_gradients) for grad in grads]
            elif clip_gradients_by == 'global_norm':
                raise NotImplementedError("Clip by global norm not implemented yet")
            else:
                raise Exception("Unkown clip gradients mode " + str(clip_gradients_by))
        sgd_step = optimizer.apply_gradients(zip(grads, vars))
        return sgd_step


def tf_inputs(input_shape, input_dtype, name, cast_to_dtype=None):
    '''returns placeholders, inputs'''
    with tf.variable_scope(name):
        placeholder = tf.placeholder(dtype=input_dtype, shape=input_shape, name="{0}_placholder".format(name))
        inputs = placeholder
        if cast_to_dtype is not None:
            inputs = tf.cast(inputs, cast_to_dtype)
        return placeholder, inputs


class TFParamsSaverLoader:
    '''Utility for Saving and loading tensorflow params'''
    def __init__(self, name, params, session: tf.Session):
        self.params = params
        self.session = session
        with tf.variable_scope(name):
            self._load_placeholders = []
            self._load_ops = []
            for p in params:
                p_placeholder = tf.placeholder(shape=p.shape.as_list(), dtype=tf.float32)
                self._load_placeholders.append(p_placeholder)
                self._load_ops.append(p.assign(p_placeholder))

    def load(self, load_path):
        loaded_params = joblib.load(load_path)
        assert len(loaded_params) == len(self._load_placeholders)
        feed_dict = {}
        for p, p_placeholder in zip(loaded_params, self._load_placeholders):
            feed_dict[p_placeholder] = p
        self.session.run(self._load_ops, feed_dict=feed_dict)

    def save(self, save_path):
        params_values = self.session.run(self.params)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(params_values, save_path)


class TFParamsCopier:
    def __init__(self, name, params_from, params_to, session: tf.Session):
        self.params_from = params_from
        self.params_to = params_to
        self.session = session
        self.ZERO_NOISE = self.generate_zero_noise()
        assert len(params_from) == len(params_to), "Number of params not same"
        if len(params_from) == 0:
            RL.logger.warn("No params to copy")
        with tf.variable_scope(name):
            self.tau, _ = tf_inputs(None, tf.float32, "tau")
            self.copy_ops = []
            self.noise_vars = []
            for p_from, p_to in zip(params_from, params_to):
                noise_var = tf.placeholder(shape=p_from.shape.as_list(), dtype=tf.float32)
                self.noise_vars.append(noise_var)
                copy_op = tf.assign(p_to, (p_from + noise_var) * self.tau + (1 - self.tau) * p_to)
                self.copy_ops.append(copy_op)

    def generate_normal_noise(self, sigma=1):
        noise = []
        for p in self.params_from:
            n = sigma * np.random.standard_normal(size=p.shape.as_list())
            noise.append(n)
        return noise

    def generate_zero_noise(self):
        noise = []
        for p in self.params_from:
            noise.append(np.zeros(p.shape.as_list()))
        return noise

    def copy(self, tau=1, noise=None):
        if noise is None:
            noise = self.ZERO_NOISE
        feed_dict = {}
        for noise_var, n in zip(self.noise_vars, noise):
            feed_dict[noise_var] = n
        feed_dict[self.tau] = tau
        self.session.run(self.copy_ops, feed_dict)


def need_conv_net(space: gym.Space):
    return isinstance(space, gym.spaces.Box) and len(space.shape) >= 2


class Model:
    """Interface for a general ML model"""

    def __init__(self):
        pass

    def predict(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self.predict(x)

    def train(self, x, targets):
        raise NotImplementedError()

    def predict_and_test(self, x, actual_y):
        raise NotImplementedError()

    def save(self, save_path):
        raise NotImplementedError()

    def load(self, load_path):
        raise NotImplementedError()


class FFNN_TF(Model):
    def __init__(self, tf_session: tf.Session, scope, input_shape, input_dtype, output_shape, hidden_size, init_scale=1.0, use_ln=False, use_bn=False, lr=1e-3, l2_reg=1e-2, clip_norm=None):
        super().__init__()
        self.session = tf_session
        with tf.variable_scope(scope):
            with tf.variable_scope('network'):
                self.inputs = tf.placeholder(
                    dtype=input_dtype, shape=[None] + input_shape, name='input')
                self.is_training = tf.placeholder(
                    dtype=tf.bool, name='is_training')
                self.outputs = tf_deep_net(
                    self.inputs, input_shape, input_dtype, 'hidden_layers_and_output', hidden_size,
                    output_shape=output_shape, init_scale=init_scale,
                    use_ln=use_ln, use_bn=use_bn, training=self.is_training)
            with tf.variable_scope('optimizer'):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
            with tf.variable_scope('optimization_ops'):
                self.trainable_vars = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
                self.targets = tf.placeholder(
                    dtype='float32', shape=[None] + output_shape)
                se = tf.square(self.outputs - self.targets)
                self.mse = tf.reduce_mean(se)
                with tf.variable_scope('L2_Losses'):
                    l2_loss = 0
                    for var in self.trainable_vars:
                        if 'bias' not in var.name and 'output' not in var.name:
                            l2_loss += l2_reg * tf.nn.l2_loss(var)
                    self.loss = self.mse + l2_loss
                update_ops = tf.get_collection(
                    tf.GraphKeys.UPDATE_OPS, scope=scope)
                with tf.control_dependencies(update_ops):
                    self.grads = tf.gradients(self.loss, self.trainable_vars)
                    if clip_norm is not None:
                        self.grads = [tf.clip_by_norm(
                            grad, clip_norm=clip_norm) for grad in self.grads]
                    self.train_op = self.optimizer.apply_gradients(
                        list(zip(self.grads, self.trainable_vars)))

            with tf.variable_scope('saving_loading_ops'):
                # for saving and loading
                self.params = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                self.load_placeholders = []
                self.load_ops = []
                for p in self.params:
                    p_placeholder = tf.placeholder(
                        shape=p.shape.as_list(), dtype=tf.float32)
                    self.load_placeholders.append(p_placeholder)
                    self.load_ops.append(p.assign(p_placeholder))

            self.writer = tf.summary.FileWriter(
                RL.logger.get_dir(), self.session.graph)

    def predict(self, x):
        return self.session.run(self.outputs, feed_dict={
            self.inputs: x,
            self.is_training: False
        })

    def __call__(self, x):
        return self.predict(x)

    def train(self, x, targets):
        '''returns mse, loss'''
        return self.session.run([self.train_op, self.mse, self.loss], feed_dict={
            self.inputs: x,
            self.targets: targets,
            self.is_training: True
        })[1:]

    def predict_and_test(self, x, actual_y):
        '''returns predicted y and mse against actual y'''
        return self.session.run([self.outputs, self.mse], feed_dict={
            self.inputs: x,
            self.is_training: False,
            self.targets: actual_y
        })

    def save(self, save_path):
        params = self.session.run(self.params)
        from baselines.a2c.utils import make_path
        make_path(os.path.dirname(save_path))
        joblib.dump(params, save_path)
        self.writer.flush()

    def load(self, load_path):
        params = joblib.load(load_path)
        feed_dict = {}
        for p, p_placeholder in zip(params, self.load_placeholders):
            feed_dict[p_placeholder] = p
        self.session.run(self.load_ops, feed_dict=feed_dict)


class FFNN_Keras(Model):
    def __init__(self, session, scope, input_shape, input_dtype, output_shape, hidden_size, init_scale=1.0, use_ln=False, use_bn=False, lr=1e-3, l2_reg=1e-2, clip_norm=None):
        super().__init__()
        if l2_reg > 0:
            raise NotImplementedError()
        if clip_norm is not None:
            raise NotImplementedError()

        model = Sequential(name=scope)
        conv_needed = len(input_shape) > 1
        if conv_needed:
            raise NotImplementedError()
        for hs in hidden_size:
            model.add(
                Dense(units=hs, input_shape=input_shape, kernel_initializer=Orthogonal(seed=0)))
            if use_ln:
                raise NotImplementedError()
            if use_bn:
                model.add(BatchNormalization())
            model.add(Activation('relu'))
        output_size = reduce(mul, output_shape, 1)
        model.add(Dense(units=output_size,
                        kernel_initializer=Orthogonal(gain=init_scale, seed=0)))
        if len(output_shape) > 1:
            raise NotImplementedError()
        optimizer = Adam(lr=lr)
        model.compile(loss='mse', optimizer=optimizer)
        self.model = model

    def predict(self, x):
        return self.model.predict(x)

    def train(self, x, targets):
        return [self.model.train_on_batch(x, targets), 0]

    def predict_and_test(self, x, actual_y):
        pred = self.model.predict(x)
        mse = self.model.evaluate(x, actual_y)
        return [pred, mse]

    def save(self, save_path):
        self.model.save_weights(save_path)

    def load(self, load_path):
        self.model.load_weights(load_path)


def StaffordRandFixedSum(n, u, nsets):

    # deal with n=1 case
    if n == 1:
        return np.tile(np.array([u]), [nsets, 1])

    k = np.floor(u)
    s = u
    step = 1 if k < (k - n + 1) else -1
    s1 = s - np.arange(k, (k - n + 1) + step, step)
    step = 1 if (k + n) < (k - n + 1) else -1
    s2 = np.arange((k + n), (k + 1) + step, step) - s

    tiny = np.finfo(float).tiny
    huge = np.finfo(float).max

    w = np.zeros((n, n + 1))
    w[0, 1] = huge
    t = np.zeros((n - 1, n))

    for i in np.arange(2, (n + 1)):
        tmp1 = w[i - 2, np.arange(1, (i + 1))] * \
            s1[np.arange(0, i)] / float(i)
        tmp2 = w[i - 2, np.arange(0, i)] * \
            s2[np.arange((n - i), n)] / float(i)
        w[i - 1, np.arange(1, (i + 1))] = tmp1 + tmp2
        tmp3 = w[i - 1, np.arange(1, (i + 1))] + tiny
        tmp4 = np.array(
            (s2[np.arange((n - i), n)] > s1[np.arange(0, i)]))
        t[i - 2, np.arange(0, i)] = (tmp2 / tmp3) * tmp4 + \
            (1 - tmp1 / tmp3) * (np.logical_not(tmp4))

    m = nsets
    x = np.zeros((n, m))
    rt = np.random.uniform(size=(n - 1, m))  # rand simplex type
    rs = np.random.uniform(size=(n - 1, m))  # rand position in simplex
    s = np.repeat(s, m)
    j = np.repeat(int(k + 1), m)
    sm = np.repeat(0, m)
    pr = np.repeat(1, m)

    for i in np.arange(n - 1, 0, -1):  # iterate through dimensions
        # decide which direction to move in this dimension (1 or 0)
        e = (rt[(n - i) - 1, ...] <= t[i - 1, j - 1])
        sx = rs[(n - i) - 1, ...] ** (1 / float(i))  # next simplex coord
        sm = sm + (1 - sx) * pr * s / float(i + 1)
        pr = sx * pr
        x[(n - i) - 1, ...] = sm + pr * e
        s = s - e
        j = j - e  # change transition table column if required

    x[n - 1, ...] = sm + pr * s

    # iterated in fixed dimension order but needs to be randomised
    # permute x row order within each column
    for i in range(0, m):
        x[..., i] = x[np.random.permutation(n), i]

    return np.transpose(x)
