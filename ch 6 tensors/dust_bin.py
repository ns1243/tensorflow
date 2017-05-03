

#예제 1
import tensorflow as tf
x = tf.constant([ [1, 2 ,3 ] ])
w = tf.constant([ [1],[2],[3] ])
y = tf.matmul(x,w)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y)
print (result)

import tensorflow as tf
def eval(a):
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    c = a.eval()
    sess.close()
    return c

x = tf.constant([ [1, 2 ,3 ] ])
w = tf.constant([ [1],[2],[3] ])
y = tf.matmul(x,w)

print (eval(y))


#예제 2
import tensorflow as tf
x = tf.Variable([ [1.,2.,3.] ], dtype=tf.float32)
w = tf.constant([ [2.],[2.],[2.]], dtype=tf.float32)
y = tf.matmul(x,w)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y)
print (result)


import tensorflow as tf
def eval(a):
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    c = a.eval()
    sess.close()
    return c

x = tf.constant((1,2,3))
w = tf.Variable([ [1],[2],[3] ])
y = tf.add(x,w)


print (type(eval(y)))





#예제 3
import tensorflow as tf
input_data = [ [1.,2.,3.],[1.,2.,3.],[2.,3.,4.] ] #3x3 matrix
x = tf.placeholder(dtype=tf.float32,shape=[None,3])
w = tf.Variable([ [2.],[2.],[2.] ], dtype = tf.float32) #3x1 matrix
y = tf.matmul(x,w)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y,feed_dict={x:input_data})
print (result)

#예제 4
import tensorflow as tf

input_data = [[1,1,1],[2,2,2]]
x = tf.placeholder(dtype=tf.float32,shape=[2,3])
w  =tf.Variable([[2],[2],[2]],dtype=tf.float32)
b  =tf.Variable([4],dtype=tf.float32)
y = tf.matmul(x,w)+b
print (x.get_shape())
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
result = sess.run(y,feed_dict={x:input_data})
print (result)