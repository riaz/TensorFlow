import tensorflow as tf

#create a constant op that produces a 1x2 matrix, the op is
# added as a node to the default graph


#the value returned by the constructor represent the output of the constant op
matrix1 = tf.constant([[3., 3.]])

#create another constant that produces a 2*1 matrix.
matrix2 = tf.constant([[2.],[2.]])

#creating a matmul operation
product = tf.matmul(matrix1,matrix2) 

if __name__ == '__main__':
    
    sess = tf.Session()
    result = sess.run(product)
    print result
    sess.close()
    

    #alternatively enter a session in block, no need to close session
    #in this way

    """
    with tf.Session() as sess:
        result = sess.run(product)
        print result
    """
