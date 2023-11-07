#coding=utf-8
#高级一点的神经网络，包含边界分析，密集连接等等
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

from test1.pub import tamPub1

# 建立模型
x=tf.placeholder(tf.float32, [None, 28 * 28], name = "x")
y=tf.placeholder(tf.float32, [None, 10], name = "y")
y_ = tf.placeholder("float", shape=[None, 10], name = "y_")
myStep = tf.placeholder(tf.float32, name = "myStep")

dnn,keep_prob = tamPub1.getMultDnn(x, [28*28, 470, 280, 10])
y_conv = tf.nn.softmax(dnn, name = "y_conv")

#w1 = weight_variable([28*28, 470], name = "w1")
#b1 = bias_variable([470], name = "b1")
#f1 = tf.nn.relu(tf.matmul(x, w1) + b1, name = "f1")
#f1 = tamPub1.getDnn(x, 28*28, 470)


#w2 = weight_variable([28*28, 500], name = "w2")
#b2 = bias_variable([500], name = "b2")
#f2 = tf.nn.relu(tf.matmul(f1, w2) + b2, name = "f2")

#w3 = weight_variable([28*28, 500], name = "w3")
#b3 = bias_variable([500], name = "b3")
#f3 = tf.nn.relu(tf.matmul(f2, w3) + b3, name = "f3")

#w3_1 = weight_variable([500, 400], name = "w3_1")
#b3_1 = bias_variable([400], name = "b3_1")
#f3_1 = tf.nn.relu(tf.matmul(f3, w3_1) + b3_1, name = "f3_1")

#w3_2 = weight_variable([400, 300], name = "w3_2")
#b3_2 = bias_variable([300], name = "b3_2")
#f3_2 = tf.nn.relu(tf.matmul(f3_1, w3_2) + b3_2, name = "f3_2")

#w4 = weight_variable([300, 150], name = "w4")
#b4 = bias_variable([150], name = "b4")
#f4 = tf.nn.relu(tf.matmul(f3_2, w4) + b4, name = "f4")

#w4 = weight_variable([470, 280], name = "w4")
#b4 = bias_variable([280], name = "b4")
#f4 = tf.nn.relu(tf.matmul(f1, w4) + b4, name = "f4")
#f4 = tamPub1.getDnn(f1, 470, 280)

#第四层，测试和训练模式切换
#keep_prob = tf.placeholder("float", name = "keep_prob")
#h_fc1_drop = tf.nn.dropout(f4, keep_prob, name = "h_fc1_drop")

#w9 = tamPub1.weight_variable([280, 10], "w9")
#b9 = tamPub1.bias_variable([10], "b9")
#y_conv = tf.nn.softmax(tf.nn.relu(tf.matmul(h_fc1_drop, w9) + b9), name = "y_conv")

#精度统计
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.maximum(y_conv, 0.01)), name = "cross_entropy")
train_step = tf.train.AdamOptimizer(myStep).minimize(cross_entropy, name = "train_step")
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1), name = "correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name = "accuracy")

#开始
from tensorflow.examples.tutorials.mnist import input_data
from time import time

#读取数据
flags = tf.app.flags
FLAGS = flags.FLAGS
#flags.DEFINE_string('data_dir', r'C:\Users\hasee\Desktop\tempdata', 'Directory for storing data') # 把数据放在/tmp/data文件夹中
flags.DEFINE_string('data_dir', r'./data/t4test', 'Directory for storing data') # 把数据放在/tmp/data文件夹中
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)   # 读取数据集

#初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
plt.figure()

writer = tf.summary.FileWriter("d://temp//tensorFlow//MnistL1",tf.get_default_graph())  
writer.close()

#循环训练
for i in range(40000):
    batch = mnist.train.next_batch(50)
    #统计并输出准确率
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0, myStep: 1e-4})
        print("step %d, training accuracy %g%%"%(i,train_accuracy*100))
        print(cross_entropy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0, myStep: 1e-4}))
        plt.subplot(3,1,1)
        if (train_accuracy == 1):
            plt.plot([i, i], [0, train_accuracy - 0.95],  color='r', linestyle='--', marker='o', label='y1 data')
        else:
            plt.plot([i, i], [0, train_accuracy - 0.95],  color='b', linestyle='--', marker='o', label='y1 data')
        plt.subplot(3,1,2)
        #plt.plot(w9.eval()[:,0],  color='r', linestyle='-', linewidth = 0.2)
        #plt.plot(w9.eval()[:,1],  color='g', linestyle='-', linewidth = 0.2)
        #plt.plot(w9.eval()[:,2],  color='b', linestyle='-', linewidth = 0.2)
        plt.subplot(3,1,3)
        #plt.plot(b9.eval(),  color='g', linestyle='-', linewidth = 0.2)
        #if (train_accuracy<0.7):
        #    print(w9.eval()[:,0])
        plt.pause(0.001)
        if (i%1000==0):
            plt.clf()
            plt.subplot(3,1,1)
            plt.plot([i + 100, i + 1000], [0, 0],  color='g', linestyle='-')
            
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, myStep: 1e-4})
    #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, myStep: 1e-4 / (i/20000*6 + 1)})

#最终准确率测试
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0, myStep: 1e-4}))

plt.show()






