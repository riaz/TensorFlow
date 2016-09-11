#Tensorflow in interactive mode / session

import tensorflow as tf

sess = tf.InteractiveSession()

x = tf.Variable([1.0,2.0])
a = tf.constant([3.0, 3.0])

#Initialize x using the run method of the initializer op.
x.initializer.run()

#adding the op to substract 'a' from x.
sub = tf.sub(x,a)
print sub.eval()

sess.close()
