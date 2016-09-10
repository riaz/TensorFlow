import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True )

#split for the mnist dataset
# Train : 55000 data points - mnist.train
# Test  : 10000 data points - mnist.test
# Valid :  5000 data points - mnist.validation

#each mnist data set contains : image and corresponding label
# eg: for the training dataset: Image: mnist.train.images (28x28) - 784 numbers
#                               Label: mnist.train.labels


x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

#model
y  = tf.nn.softmax(tf.matmul(x,W) + b)

#implementing cross-entropy
y_ = tf.placeholder(tf.float32, [None,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

if __name__ == '__main__':
    
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
