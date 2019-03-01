import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data_x = np.float32(np.random.rand(2, 100))  # 生成一个2*100 维的数据矩阵
label_y = np.dot([0.100, 0.200], data_x) + 0.3  # 生成一个1 *100的数据

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, 2]))
b = tf.Variable(tf.random_uniform([1]))

init = tf.global_variables_initializer()

predict = tf.matmul(W, x) + b

loss = tf.reduce_mean(tf.square(y - predict))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

predict_y = None
with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        sess.run(optimizer, feed_dict={x: data_x, y: label_y})
        if i % 20 == 0:
            print(sess.run(W), sess.run(b), )

    plt.scatter(data_x[0, :], label_y)
    predict_y = sess.run(predict, feed_dict={x: data_x})
    plt.plot(data_x[0, :], data_x[0, :], color="red")
plt.show()
