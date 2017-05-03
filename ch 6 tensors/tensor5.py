#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def eval(a):
    sess = tf.InteractiveSession()
    c = a.eval()
    sess.close()
    return c
    

    
# 예제 16
print ("예제 16")

a = np.arange(6)
a = a * 2

print ("a is \n", a)

a_max = np.argmax(a)
a_min = np.argmin(a)

print ("a_max \n", a_max)
print ("a_min \n", a_min)

b = np.reshape(a, [2,3])


print ("b is \n", b)
b_max = np.argmax(b, axis=0)
b_min = np.argmin(b, axis=1)
print ("b_max \n", b_max)
print ("b_min \n", b_min)

# 예제 17
print ("예제 17")

a = np.arange(12)
a = a * 2

print ("a is \n", a)

tf_a = tf.constant(a, tf.float32)

a_reduce_mean = tf.reduce_mean(tf_a) 
print ("a_reduce_mean is ", eval(a_reduce_mean))



# 예제 18
print ("예제 18")

a = np.arange(6)
a = a * 2
a = np.reshape(a,[2,3])
print ("a is \n", a)

tf_a = tf.constant(a, tf.float32)

a_reduce_mean0 = tf.reduce_mean(tf_a, axis=0) 
print ("a_reduce_mean axis 0 is \n", eval(a_reduce_mean0))

a_reduce_sum0 = tf.reduce_sum(tf_a, axis=0)
print ("a_reduce_sum axis 0 is \n", eval(a_reduce_sum0))

a_reduce_mean1 = tf.reduce_mean(tf_a, axis=1) 
print ("a_reduce_mean axis 1 is \n", eval(a_reduce_mean1))

a_reduce_sum1 = tf.reduce_sum(tf_a, axis =1)
print ("a_reduce_sum axis 1 is \n", eval(a_reduce_sum1))


# 예제 19
print ("예제 19")

a = np.arange(12)
a = a * 2
a = np.reshape(a,[2,2,3])
print ("a is \n", a)

tf_a = tf.constant(a, tf.float32)


a_reduce_mean0 = tf.reduce_mean(tf_a, axis=0) 
print ("a_reduce_mean axis 0 is \n", eval(a_reduce_mean0))

a_reduce_sum0 = tf.reduce_sum(tf_a, axis=0)
print ("a_reduce_sum axis 0 is \n", eval(a_reduce_sum0))

a_reduce_mean1 = tf.reduce_mean(tf_a, axis=1) 
print ("a_reduce_mean axis 1 is \n", eval(a_reduce_mean1))

a_reduce_sum1 = tf.reduce_sum(tf_a, axis =1)
print ("a_reduce_sum axis 1 is \n", eval(a_reduce_sum1))

a_reduce_mean2 = tf.reduce_mean(tf_a, axis=2) 
print ("a_reduce_mean axis 2 is \n", eval(a_reduce_mean2))

a_reduce_sum2 = tf.reduce_sum(tf_a, axis=2) 
print ("a_reduce_sum axis 2 is \n", eval(a_reduce_sum2))

