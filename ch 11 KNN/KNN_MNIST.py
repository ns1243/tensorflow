#-*- coding: utf-8 -*-
# 12장 교재용 KNN.py에서 변형되었습니다.
# test를 전체 다 검증하는게 아니라 랜덤으로 선정해서
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from random import choice, shuffle
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# x_data = mnist.train.images
# y_data = mnist.train.labels
x_data2 = mnist.test.images
y_data2 = mnist.test.labels

x_data, y_data = mnist.train.next_batch(55000)

#MNIST data 처리
#
#
# data_file_name= 'kNN.txt' # 데이터 파일 이름을 변수에 입력한다.
# xy=np.genfromtxt(data_file_name,dtype='float32') #파일의 데이터를 불러 들인다.
#
# data_file_name2= 'k_NN_test.txt' # 데이터 파일 이름을 변수에 입력한다.
# xy2=np.genfromtxt(data_file_name2,dtype='float32') #파일의 데이터를 불러 들인다.
#

number_of_test = 100
count = len(x_data2)
indices = list(range(count))
shuffle(indices)



#shuffle(xy)

# # x값과 y값을 불러 온다.
# x_data = xy[: , 0:2] # 첫째 둘째 칼럼을 x_data로 저장
# y_data = xy[: , 2:5] # 둘째 칼럼을 y_data로 저장
#
# x_data2 = xy2[:, 0:2]
# y_data2 = xy2[:, 2:5]
# plt.plot(x_data, y_data, 'ro', alpha = 0.3)
# plt.legend()
# plt.show()
#
# number_of_cluster = 5
# count = len(xy)
#
#
#
# centroid_x = []
# centroid_y = []
# indices = list(range(len(xy)))
# shuffle(indices)
# for i in range(0,number_of_cluster):
#     centroid_x.append(x_data[indices[i]])
#     centroid_y.append(y_data[indices[i]])
#



# print ("centroid x", centroid_x)
# print ("centroid y", centroid_y)
#

x_train = tf.constant(x_data)
y_train = tf.constant(y_data)
# y_points = tf.constant(y_data)
x_test = tf.placeholder(tf.float32,[784])


#expanded_x_train = tf.expand_dims(x_train, 0)
#expanded_x_test  = tf.expand_dims(x_test, 1)
reshape_x_train = tf.reshape(x_train,[1,-1,784])
reshape_x_test = tf.reshape(x_test, [1,1,-1])
tf_sub = tf.sub(reshape_x_train,x_test)

distances_array =  tf.sqrt(tf.reduce_sum(tf.square(tf_sub),2))
assignments = tf.arg_min(distances_array,1)
# y = tf.argmax(y_train,1)
# guess = tf.argmax(y, 1)
# ========================================================  2부


sess = tf.Session()
init_op = tf.initialize_all_variables()
sess.run(init_op)

correct_prediction = 0
false_prediction = 0
for step in range(number_of_test):
    index = indices[step]
    feed = {x_test: x_data2[index]}
    distances_out = sess.run(distances_array, feed_dict=feed)
    assignments_out = sess.run(assignments, feed_dict=feed)
    guess = y_data[assignments_out]
    y = y_data2[index]
    t_sub_out, x_test_out, x_train_out = sess.run([tf_sub,x_test, x_train], feed_dict=feed)

    a = 1
    #if ([y] == guess):
    if(np.argmax(y,0) == np.argmax(guess,1)):
        correct_prediction = correct_prediction + 1
    else:
        false_prediction = false_prediction +1

    accuracy = 100 * correct_prediction/number_of_test
print ("test ended")
print ("correct prediction rate is ", accuracy, " %")

