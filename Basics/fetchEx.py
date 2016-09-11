#Using fetch to obtain the output from multiple ops

import tensorflow as tf

input1 = tf.constant([3.0])
input2 = tf.constant([2.0])
input3 = tf.constant([5.0])

intermed = tf.add(input2,input3)

res = tf.mul(input1,intermed)

with tf.Session() as sess:
    result = sess.run([res,intermed]) #this is where fetch retrieve 2 ops
    print result

