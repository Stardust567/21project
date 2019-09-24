import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# 占位符
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# 变量
W = tf.Variable(tf.zeros([784, 10]))
tf.Variable()
b = tf.Variable(tf.zeros([10]))
# 计算
y = tf.nn.softmax(tf.matmul(x, W)+b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# 会话
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 运行
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
# 检测
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# bool -> float
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))
