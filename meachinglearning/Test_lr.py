import tensorflow as tf
import numpy as np

# 数据集
x_data = np.float32(np.random.rand(2, 100))  # 生成一个2*100 维的数据矩阵
y_data = np.dot([0.100, 0.200], x_data) + 0.3  # 生成一个1 *100的数据

# 输入变量
input_x = tf.placeholder(tf.float32, [2, 100])
output_y = tf.placeholder(tf.float32, [100])

# 构建一个线性模型

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2]))
predict_y = tf.matmul(W, input_x) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(output_y - predict_y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.2).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 拟合平面
with tf.Session() as sess:
    sess.run(init)
    for i in range(2000):
        sess.run(optimizer, feed_dict={input_x: x_data, output_y: y_data})
        if i % 20 == 0:
            print(i, sess.run(W), sess.run(b))
