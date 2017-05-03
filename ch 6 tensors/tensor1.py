
#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


# 예제 1
a = 1
b = [1]
c = [[[1]]]

print ("a", type(a))
print ("b", type(b))
print ("c", type(c))


d = a+ b + c
e = b + c
print ("list d is", d)
print ("list e is ", e)

# 예제 2
a1 = np.array(1)
b1 = np.array([1])
c1 = np.array([[[1]]])

print ("a1", type(a1))
print ("b1", type(b1))
print ("c1", type(c1))
print ("c1 shape", np.shape(c1))
d1=  b1 + c1
print ("numpy array d1 is ",d1)

# 예제 3
a1 = np.array(1)
b1 = np.array([1])
c1 = np.array([[[1]]])
d1= a1 + b1 + c1
print ("numpy array d1 is ",d1)

#예제 3
a = ["a", "b", "c"]

print (" a type is ", type(a))

print (a[0])
print (a[1])
print (a[2])

a[2] = "ccc"
print (a)

print (type(a[1]))