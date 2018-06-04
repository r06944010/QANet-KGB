import tensorflow as tf


def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(input, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)

def my_image_filter(input_images):
    with tf.variable_scope("conv1", reuse = tf.AUTO_REUSE):
        # Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu(input_images, [5, 5, 32, 32], [32])
    with tf.variable_scope("conv2", reuse = tf.AUTO_REUSE):
        # Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu(relu1, [5, 5, 32, 32], [32])

with tf.variable_scope("image_filters", reuse = tf.AUTO_REUSE) as scope:
    image1 = tf.get_variable('image1', [10,32,32,32])
    image2 = tf.get_variable('image2', [10,32,32,32])
    result1 = my_image_filter(image1)
    scope.reuse_variables()
    result2 = my_image_filter(image2)
    print(image1)
    print(image2)
    print(image1 == image2)
    print(result1)
    print(result2)
    print(result1 == result2)
with tf.variable_scope("image_filters", reuse = True):
    image3 = tf.get_variable('image1', [10,32,32,32])
    print(image3)
    print(image1 == image3)
with tf.Session() as sess:
    writer = tf.summary.FileWriter('test')
    writer.add_graph(sess.graph)
    writer.flush()
'''
import tensorflow as tf
import numpy as np


x = tf.constant(np.random.uniform(-3, 3, 10)) 
y = tf.constant(np.random.uniform(-1, 1, 10))
s = tf.losses.cosine_distance(tf.nn.l2_normalize(x, 0), tf.nn.l2_normalize(y, 0), dim=0)
#s = tf.losses.cosine_distance(x, y, dim=0)
print(tf.Session().run(s))
'''
