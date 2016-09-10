import tensorflow as tf
import numpy as np

#Create 100 phony x,y data points in NumPy,y = x*0.1 + 0.3

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

#consider the y_data as y = W * x +b
#given the x and the y information , we make tensor flow to compute the valie
#of W and b for us, which is 0.1 and 0.3 in the above case
#this is the first exposure to the tensor flow library

W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

#Minimize the mean square errors & applying gradient descent to minimize the loss
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#before starting, we will initialize the variables
# we will run this first

init = tf.initialize_all_variables()

#Launch the graph
sess = tf.Session()
sess.run(init)

#Fit the line. since we are using gradient descent
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print (step, sess.run(W), sess.run(b) )

#Learns best fit is W: [0.1] , b: [0.3]

