#coding=utf-8
#第一个股票预测分析系统
analyzeZone = 50; #分析区间

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

filename = "./stock/600019.csv"
print("=======>> 开始载入股票数据：%s"%(filename))
csv_reader = csv.reader(open(filename, encoding="gb2312"))
#开盘，收盘，最高，最低，换手率
stockData =  np.empty(shape=[0, 5])
i = 0
for row in csv_reader:
    if i > 1: #跳过题头和开盘第一天
        stockData = np.vstack((stockData, [float(row[1])/10, float(row[4])/10, float(row[2])/10, float(row[3])/10, float(row[9])/100]))
    i = i + 1
#print(np.append(np.transpose(sOpen, 0), sClose.transpose()))
print("=======>> 股票数据载入完毕，共载入%d条记录"%(len(stockData)))

#随机正态分布
def weight_variable(shape, name = "variable"):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)

#常量
def bias_variable(shape, name = "variable"):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

#卷积
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#池化
def max_pool_2x2(x, name = "variable"):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = name)

stockDataLen = len(stockData)
stockData = np.reshape(stockData, (len(stockData)*5))
print(stockData)
    
#创建模型
#y = tf.placeholder(tf.float32, [4], name = "y") #开盘，收盘，最高，最低
x = tf.placeholder(tf.float32, shape = [None, analyzeZone * 5], name = "x")
y_ = tf.placeholder(tf.float32, shape = [None, 1], name = "y_") #开盘，收盘，最高，最低
myStep = tf.placeholder(tf.float32, name = "myStep")

w1 = weight_variable([analyzeZone * 5, 70], name = "w1")
b1 = bias_variable([70], name = "b1")
f1 = tf.nn.relu(tf.matmul(x, w1) + b1, name = "f1")
#f1 = tf.matmul(x, w1) + b1

w4 = weight_variable([70, 30], name = "w4")
b4 = bias_variable([30], name = "b4")
f4 = tf.nn.relu(tf.matmul(f1, w4) + b4, name = "f4")
#f4 = tf.matmul(f1, w4) + b4

#第四层，测试和训练模式切换
keep_prob = tf.placeholder("float", name = "keep_prob")
h_fc1_drop = tf.nn.dropout(f4, keep_prob, name = "h_fc1_drop")

w9 = weight_variable([30, 1], "w9")
b9 = bias_variable([1], "b9")
y_conv = tf.nn.relu(tf.matmul(h_fc1_drop, w9) + b9)
#y_conv = tf.matmul(h_fc1_drop, w9) + b9

#精度统计
accuracy = (y_[0,0] - y_conv[0,0])/y_[0,0]
#cross_entropy = tf.reduce_sum((y_ - y_conv)*(y_ - y_conv))
cross_entropy = accuracy*accuracy*100000
train_step = tf.train.AdamOptimizer(myStep).minimize(cross_entropy, name = "train_step")
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1), name = "correct_prediction")
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name = "accuracy")

#开始
from time import time

#初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
plt.figure()

print("start")

#循环训练
repeat = 3
i = 0
while i < stockDataLen - analyzeZone - 2:
    batch = stockData[i * 5 : (i + analyzeZone) * 5]
    resultTrain = [[
        (stockData[(i + analyzeZone) * 5 + 0] +
        stockData[(i + analyzeZone) * 5 + 1]) / 2
        ]]
    train_step.run(feed_dict={x: np.reshape(batch, (1, analyzeZone * 5)), y_: resultTrain, keep_prob: 0.5, myStep: 1e-4})
    #result = y_conv.eval(feed_dict={x: np.reshape(batch, (1, analyzeZone * 5)), y_: resultTrain, keep_prob: 1.0, myStep: 1e-4})
    #print("==============")
    #print(resultTrain)
    #print(result)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: np.reshape(batch, (1, analyzeZone * 5)), y_: resultTrain, keep_prob: 1.0, myStep: 1e-4})
        result = y_conv.eval(feed_dict={x: np.reshape(batch, (1, analyzeZone * 5)), y_: resultTrain, keep_prob: 1.0, myStep: 1e-4})
        d1 = (stockData[(i + analyzeZone - 1) * 5 + 0] +
            stockData[(i + analyzeZone - 1) * 5 + 1]) / 2
        print("%d ==> percent: %f%%, 实际:%f 预测:%f"%(i, train_accuracy*100, (resultTrain[0][0] - d1)*10, (result[0,0] - d1)*10))
    #print(cross_entropy.eval(feed_dict={x: np.reshape(batch, (1, analyzeZone * 5)), y_: resultTrain, keep_prob: 1.0, myStep: 1e-4}))
    #print(resultTrain)
    #print(y_conv.eval(feed_dict={x: np.reshape(batch, (1, analyzeZone * 5)), y_: resultTrain, keep_prob: 1.0, myStep: 1e-4}))
    i = i + 1
    if i == stockDataLen - analyzeZone - 2 - 1:
        print("======================================================")
        repeat = repeat - 1
        if repeat == 0: break
        i = 0

#plt.show()



























    
    





