# import tensorflow as tf
# import sys
# from enum import Enum
# from RL.common.utils import auto_conv_dense_net


# class Context:
#     convs = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
#     deconvs = []
#     hidden_layers = []
#     embedding_size = 256

#     def __init__(self):
#         self.session = tf.Session()

#     def _read_args(self):
#         for arg in sys.argv[1:]:
#             words = arg.split('=')
#             assert len(
#                 words) == 2, "Invalid argument {0}. The format is --arg_name=arg_value".format(arg)
#             assert words[0].startswith(
#                 '--'), "Invalid argument {0}. The format is --arg_name=arg_value".format(arg)
#             arg_name = words[0][2:]
#             try:
#                 arg_value = eval(words[1])
#             except Exception:
#                 arg_value = words[1]
#             assert hasattr(self, arg_name), "Unrecognized argument {0}".format(arg_name)
#             setattr(self, arg_name, arg_value)
    
#     @property
#     def need_conv_net(self):
#         return True

#     def close(self):
#         self.session.close()

#     def activation_fn(self, x):
#         return tf.nn.relu(x)


# class GAN:
#     class Scopes(Enum):
#         generator = "Generator"
#         discriminator = "Discriminator"

#     def __init__(self, context: Context, name, actuals):
#         self.fakes = []
#         self.actuals = actuals
#         self.context = context
#         self.name = name
#         with tf.name_scope(self.name):
#             self._discriminator = self.tf_discriminator()

#     def tf_generator(self, z):
#         with tf.variable_scope(GAN.Scopes.generator.value, reuse=tf.AUTO_REUSE):
#             h = tf.layers.dense()
#             dc1 = tf.layers.conv2d_transpose(z, )

#     def tf_discriminator(self, inputs):
#         with tf.variable_scope(GAN.Scopes.discriminator.value, reuse=tf.AUTO_REUSE):
#             return auto_conv_dense_net(self.context.need_conv_net, inputs, self.context.convs, self.context.hidden_layers + self.context.embedding_size, self.context.activation_fn, 1, tf.nn.tanh, "auto_conv_dense_net")
    
""" Deep Convolutional Generative Adversarial Network (DCGAN).
Using deep convolutional generative adversarial networks (DCGAN) to generate
digit images from a noise distribution.
References:
    - Unsupervised representation learning with deep convolutional generative
    adversarial networks. A Radford, L Metz, S Chintala. arXiv:1511.06434.
Links:
    - [DCGAN Paper](https://arxiv.org/abs/1511.06434).
    - [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
from RL.common.plot_renderer import PlotRenderer
from RL.common.utils import GridImageViewer, SimpleImageViewer
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Params
num_steps = 20000
batch_size = 32

# Network Params
image_dim = 784 # 28*28 pixels * 1 channel
gen_hidden_dim = 256
disc_hidden_dim = 256
noise_dim = 200 # Noise data points


# Generator Network
# Input: Noise, Output: Image
def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        # TensorFlow Layers automatically create variables and calculate their
        # shape, based on the input.
        x = tf.layers.dense(x, units=6 * 6 * 128)
        x = tf.nn.tanh(x)
        # Reshape to a 4-D array of images: (batch, height, width, channels)
        # New shape: (batch, 6, 6, 128)
        x = tf.reshape(x, shape=[-1, 6, 6, 128])
        # Deconvolution, image shape: (batch, 14, 14, 64)
        x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)
        # Deconvolution, image shape: (batch, 28, 28, 1)
        x = tf.layers.conv2d_transpose(x, 1, 2, strides=2)
        # Apply sigmoid to clip values between 0 and 1
        x = tf.nn.sigmoid(x)
        return x


# Discriminator Network
# Input: Image, Output: Prediction Real/Fake Image
def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        # Typical convolutional neural network to classify images.
        x = tf.layers.conv2d(x, 64, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 128, 5)
        x = tf.nn.tanh(x)
        x = tf.layers.average_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 1024)
        x = tf.nn.tanh(x)
        # Output 2 classes: Real and Fake images
        x = tf.layers.dense(x, 2)
    return x

# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

# Build Generator Network
gen_sample = generator(noise_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)
disc_concat = tf.concat([disc_real, disc_fake], axis=0)

# Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Targets (real or fake images)
disc_target = tf.placeholder(tf.int32, shape=[None])
gen_target = tf.placeholder(tf.int32, shape=[None])

# Build Loss
disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_concat, labels=disc_target))
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=gen_target))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.001)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=0.001)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Create training operations
train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # f, a = plt.subplots(4, 10, figsize=(10, 4))
    visual_z = np.random.uniform(-1., 1., size=[10, 4, noise_dim])
    grid = GridImageViewer(4, 10)
    img_list = []
    plot = PlotRenderer(title="GAN Losses", xlabel="step", ylabel="cross entropy loss", smoothing=10)
    plot.plot([], [], [], [])
    plot.axes.legend(["Generator", "Discriminator"])

    for step in range(1, num_steps + 1):

        # Prepare Input Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, newshape=[-1, 28, 28, 1])
        # Generate noise to feed to the generator
        z = np.random.uniform(-1., 1., size=[batch_size, noise_dim])

        # Prepare Targets (Real image: 1, Fake image: 0)
        # The first half of data fed to the generator are real images,
        # the other half are fake images (coming from the generator).
        batch_disc_y = np.concatenate(
            [np.ones([batch_size]), np.zeros([batch_size])], axis=0)
        # Generator tries to fool the discriminator, thus targets are 1.
        batch_gen_y = np.ones([batch_size])

        # Training
        feed_dict = {real_image_input: batch_x, noise_input: z,
                     disc_target: batch_disc_y, gen_target: batch_gen_y}
        _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss],
                                feed_dict=feed_dict)
        plot.append([gl, dl], starting_x=1)
        if step % 100 == 0 or step == 1:
            print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (step, gl, dl))
        if step % 10 == 0 or step == 1:
            img_list.clear()
            for i in range(10):
                # Noise input.
                # z = np.random.uniform(-1., 1., size=[4, noise_dim])
                g = 255 * sess.run(gen_sample, feed_dict={noise_input: visual_z[i]})
                for j in range(4):
                    # Generate image from noise. Extend to 3 channels for matplot figure.
                    img = np.reshape(np.repeat(g[j][:, :, np.newaxis], 3, axis=2), newshape=(28, 28, 3)).astype(np.uint8)
                    # img = np.expand_dims(g[j][], 2)
                    # img = np.concatenate([img] * 3, axis=2)
                    # a[j][i].imshow(img)
                    img_list.append(img)
            # f.show()
            # plt.draw()
            # plt.show(block=False)
            # plt.pause(0.01)
            plot.render()
            grid.imshow(img_list)
        grid.dispatch_events()
        plot.dispatch_events()
