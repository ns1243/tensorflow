
#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf



# 예제 4
a = tf.constant(1)
b = tf.constant([1])
c = tf.constant([[[1]]])
print ("a", type(a))
print ("b", type(b))
print ("c", type(c))

d = a + b + c
print (" a + b + c is ", d)




# 예제 5
a = tf.constant(1)
b = tf.constant([1])
c = tf.constant([[[1]]])

sess = tf.InteractiveSession()
d = a + b + c
print (" a + b + c is ", d.eval())


# 예제 6
def eval(a):
    sess = tf.InteractiveSession()
    c = a.eval()
    sess.close()
    return c


a = tf.constant(1)
b = tf.constant([1])
c = tf.constant([[[1]]])


d = a + b + c
print (" a + b + c is ", eval(d))
