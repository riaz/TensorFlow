import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_from_sets("MNIST_data",one_hot=True)

ses = tf.InteractiveSession()

#2d tensor of image pixels
x  = tf.placeholder(tf.float32, shape=[ None, 784 ])

#target output
y_ = tf.placeholder(tf.float32, shape=[None, 10])


#defining weights and bias for the model

W = tf.Variable(tf.zeros([784,10])) 
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

#regression model

y = tf.nn.softmax(tf.matmul(x,W) + b)

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

