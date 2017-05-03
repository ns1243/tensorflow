#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def eval(a):
    sess = tf.InteractiveSession()
    c = a.eval()
    sess.close()
    return c
    
    

# 예제 10
print ("예제 10 행렬곱")

a = np.arange(6)
a = np.reshape(a, [2,3])

b = np.arange(6)
b = np.reshape(b, [2,3])
print ("a is \n", a)
print ("b is \n", b)

# tensor에 입력하기 
tf_a = tf.constant(a, dtype=tf.float32)
tf_b = tf.constant(b, dtype=tf.float32)
tf_c = tf.matmul(a,b)
#행렬곱
print ("matrix multiplying  \n", eval(tf_c))


# 예제 11
print ("예제 11")

a = np.arange(6)
a = np.reshape(a, [2,3])
print ("a is \n", a)
b = np.array(3)
print ("b is \n", b)
print ("a + b  is \n", a + b)


# 예제 12
print ("예제 12")

a = np.arange(6)
a = np.reshape(a, [2,3])
tf_a = tf.constant(a, tf.float32)

print ("tf_a is \n", eval(tf_a))

b = np.array([1,2,3])
tf_b = tf.constant(b, tf.float32)

print ("tf_b is \n", eval(tf_b))


print ("tf_a + tf_b  is \n", eval(tf_a + tf_b))

# 예제 13
print ("예제 13")

tf_a = tf.zeros([255,255,2], tf.float32)
tf_b = tf.zeros([2], tf.float32)
tf_c = tf_a + tf_b
print ("successful 1  \n", tf_c)

tf_a = tf.zeros([255,255,2], tf.float32)
tf_b = tf.zeros([1,2], tf.float32)
tf_c = tf_a + tf_b
print ("successful 2 \n", tf_c)

tf_a = tf.zeros([255,255,2], tf.float32)
tf_b = tf.zeros([1,1,2], tf.float32)
tf_c = tf_a + tf_b
print ("successful 3 \n", tf_c)


# 예제 14
print ("예제 14")
tf_a = tf.zeros([255,1,2], tf.float32)
tf_b = tf.zeros([1,255,1], tf.float32)
tf_c = tf_a + tf_b
print ("successful 4 \n", tf_c)


tf_a = tf.zeros([1024,768], tf.float32)
tf_b = tf.zeros([1,768], tf.float32)
tf_c = tf_a + tf_b
print ("successful 5 \n", tf_c)

tf_a = tf.zeros([1,2,3,4], tf.float32)
tf_b = tf.zeros([2,1,1,4], tf.float32)
tf_c = tf_a + tf_b
print ("successful 6 \n", tf_c)



# 예제 15
print ("예제 15")
tf_a = tf.zeros([200,255], tf.float32)
tf_b = tf.zeros([2], tf.float32)
tf_a_reshape = tf.reshape(tf_a, [1,200,255])
tf_b_reshape = tf.reshape(tf_b, [2,1,1])
tf_c = tf_a_reshape + tf_b_reshape
print ("successful 7 \n", tf_c)


tf_a = tf.zeros([200,255], tf.float32)
tf_b = tf.zeros([1,2], tf.float32)
tf_a_reshape = tf.reshape(tf_a, [200,255,1])
tf_b_reshape = tf.reshape(tf_b, [1,1,2])
tf_c = tf_a_reshape + tf_b_reshape
print ("successful 8 \n", tf_c)

tf_a = tf.zeros([200,255], tf.float32)
tf_b = tf.zeros([200,2], tf.float32)
tf_a_reshape = tf.reshape(tf_a, [200,255,1])
tf_b_reshape = tf.reshape(tf_b, [200,1,2])
tf_c = tf_a_reshape + tf_b_reshape
print ("successful 9 \n", tf_c)



