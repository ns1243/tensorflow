#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def eval(a):
    sess = tf.InteractiveSession()
    c = a.eval()
    sess.close()
    return c


# 예제 7
print ("예제 7")
a = np.arange(6)
print ("first \n",a)
a = np.reshape(a, [2,3])
print ("second \n",a)
a = np.reshape(a, [3,2])
print ("third \n",a)


# 예제 8
print ("예제 8")
a = np.arange(20)
print ("first \n",a)
a = np.reshape(a, [2,2,5])
print ("second \n",a)
a = np.reshape(a, [2,2,-1])
print ("third \n",a)
a = np.reshape(a, [2,-1,5])
print ("fourth \n",a)


# 예제 9
print ("예제 9")
a = np.arange(6)
a = np.reshape(a, [2,3])

b = np.ones(6)
b = np.reshape(b, [2,3])
print ("a is \n", a)
print ("b is \n", b)
# tensor에 입력하기 
tf_a = tf.constant(a, dtype=tf.float32)
tf_b = tf.constant(b, dtype=tf.float32)

print ("adding  \n",eval(tf_a + tf_b))
print ("subtracting  \n",eval(tf_a - tf_b))
print ("multiplying \n", eval(tf_a * tf_b))
print ("dividing  \n", eval(tf_a / tf_b))


# 예제 9-1
print ("예제 9-1 tensorflow 함수 쓰기")
a = np.arange(6)
a = np.reshape(a, [2,3])

b = np.ones(6)
b = np.reshape(b, [2,3])
print ("a is \n", a)
print ("b is \n", b)
# tensor에 입력하기 
tf_a = tf.constant(a, dtype=tf.float32)
tf_b = tf.constant(b, dtype=tf.float32)

print ("adding  \n",eval(tf.add(tf_a, tf_b)))
print ("subtracting  \n",eval(tf.sub(tf_a, tf_b)))
print ("multiplying \n", eval(tf.multiply(tf_a , tf_b)))
print ("dividing  \n", eval(tf.divide(tf_a , tf_b)))
