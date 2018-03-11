import tensorflow as tf
import sys
from os import path
cur_file = path.dirname(__file__)
sys.path.append(cur_file)
import input_data

# None 表示第一维可以是任意长，因为我们还不知道样本的总数量。而样本的特征维数是784
x = tf.placeholder("float", [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

mnist = input_data.read_data_sets('/data/code/github/machine_learn_msw/tensorflow/mnist/mnist_data', one_hot=True)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# 1. 构建输入输出 - placeholder
# 2. 构建计算图
# 3. 定义损失函数
# 4. 最小化/优化 损失函数
# 5. 执行session，执行训练

