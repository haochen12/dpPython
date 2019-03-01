import os
import struct
import numpy as np


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


mnist, lable = load_mnist('data')

# print(np.shape(mnist))
# print(np.shape(lable)[0])

# # print(lable)
import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 784], "X")
Y = tf.placeholder(tf.float32, [None, 10], "Y")

# 定义模型参数
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10, ]))

output = tf.nn.xw_plus_b(X, W, b)

# prob = tf.nn.softmax(output)
#
# loss = tf.square(Y - output)
#
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    print(sess.run(output))
    # for i in range(10):
    for x, y in zip(mnist, lable):
        print(sess.run(output, feed_dict={X: x, Y: y}))
# w_out, b_out = sess.run([W, b])
# print(b_out)
