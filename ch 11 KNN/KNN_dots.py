#-*- coding: utf-8 -*-
# 12장 교재용 KNN.py에서 변형되었습니다.
# test를 전체 다 검증하는게 아니라 랜덤으로 선정해서
import numpy as np
import tensorflow as tf
from random import shuffle

data_file_name= 'label_dots.txt' # 데이터 파일 이름을 변수에 입력한다.
xy=np.genfromtxt(data_file_name,dtype='float32') #파일의 데이터를 불러 들인다.

data_file_name2= 'test_dots.txt' # 데이터 파일 이름을 변수에 입력한다.
xy2=np.genfromtxt(data_file_name2,dtype='float32') #파일의 데이터를 불러 들인다.


number_of_test = 70
count = len(xy2)
indices = list(range(count))
shuffle(indices)



#shuffle(xy)

# x값과 y값을 불러 온다.
x_data = xy[: , 0:2] # 첫째 둘째 칼럼을 x_data로 저장
y_data = xy[: , 2:5]

x_data2 = xy2[:, 0:2]
y_data2 = xy2[:, 2:5]

x_train = tf.constant(x_data)
y_train = tf.constant(y_data)
# y_points = tf.constant(y_data)
x_test = tf.placeholder(tf.float32,[2])
reshape_x_train = tf.reshape(x_train,[1,-1,2])
reshape_x_test = tf.reshape(x_test, [1,-1,2])
tf_sub = tf.sub(reshape_x_train,reshape_x_test)

distances_array =  tf.reduce_sum(tf.square(tf_sub),2)
assignments = tf.arg_min(distances_array,1)

# ========================================================  2부

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

correct_prediction = 0

for step in range(number_of_test):
    index = indices[step]
    feed = {x_test: x_data2[index]}
    distances_out = sess.run(distances_array, feed_dict=feed)
    assignments_out = sess.run(assignments, feed_dict=feed)
    guess = y_data[assignments_out]
    y = y_data2[index]
    t_sub_out, x_test_out, x_train_out = sess.run([tf_sub,x_test, x_train], feed_dict=feed)

    a = 1
    if (np.argmax(y,0) == np.argmax(guess,1)):
        correct_prediction = correct_prediction + 1
print ("test ended")
print ("correct prediction rate is ", 100* correct_prediction/number_of_test)
