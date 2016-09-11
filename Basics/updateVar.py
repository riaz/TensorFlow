#understanding variables in tensorflow

import tensorflow as tf

#Creating a varibale that will be initialized to the scalar value 0
state = tf.Variable(0, name = "counter")

#creating a op to add one to state
one = tf.constant(1)
new_value = tf.add(state,one)

update = tf.assign(state,new_value)

#the variables must be initialized by running a init.

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(state)) #prints the initial value of state

    #run the op that updates state and print state

    for _ in range(3):
        sess.run(update)
        print (sess.run(state))


    
    
