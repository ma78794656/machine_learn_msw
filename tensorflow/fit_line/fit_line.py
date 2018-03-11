import tensorflow as tf
import numpy as np

# fake test data and result
x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

# construct TF variables
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# min the MSE
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# initialize the variable
init = tf.initialize_all_variables()

# launch the graph
sess = tf.Session()
sess.run(init)

for step in range(1, 201):
    sess.run((train))
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))
