import tensorflow as tf
import test.input_data as input_data

mnist = input_data.read_data_sets("./data", one_hot=True)
x = tf.placeholder("float", [None, 784])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, w) + b)

y_ = tf.placeholder("float", [None, 10]);

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
cross_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuary = tf.reduce_mean(tf.cast(cross_prediction, "float"))
print(sess.run(accuary, {x: mnist.test.images, y_: mnist.test.labels}))
