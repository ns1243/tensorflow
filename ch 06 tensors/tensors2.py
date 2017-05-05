#-*- coding: utf-8 -*-





#예제 7

import numpy as np
a = [ [1 , 2], [3 , 4] ]
b = [ [1 , 2], [3 , 4] ]
mul = np.multiply(a , b)
mat_mul = np.matmul(a , b)
print("mul ", mul)
print("matmul ", mat_mul)


#예제 8
import tensorflow as tf

data = [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
a = tf.placeholder(dtype=tf.float32,shape=[None,3])
b  =tf.Variable([0,1,2,4],dtype=tf.float32)
b  = tf.reshape(b, [4, -1])
y = tf.add(a,b)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
y_out = sess.run(y,feed_dict={a:data})
print (y_out)












import tensorflow as tf

data = [[1],[2],[3],[4]]
print ("data type is", type(data))
a = tf.placeholder(dtype=tf.float32,shape=[None,None])

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
a_out = sess.run(a,feed_dict={a:data})

print ("a type is", type( a_out))

