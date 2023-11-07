#coding=utf-8
#第二个股票预测分析系统
analyzeZone = [100, 50, 25, 10, 5, 3, 2, 1]; #分析区间
analyzeL1 =   [200, 100,  50, 20, 10,  6, 10, 5]; #第一步特征
analyzeL2 =   [50,   25,   25,  10,  5,  5, 5, 5]; #第二步特征

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

filename = "./stock/600019.csv"
print("=======>> 开始载入股票数据：%s"%(filename))
csv_reader = csv.reader(open(filename, encoding="gb2312"))
#开盘，收盘，最高，最低，换手率
stockData =  np.empty(shape=[0, 5])
stockNote =  np.empty(shape=[0, 2])
i = 0
for row in csv_reader:
    if i > 1: #跳过题头和开盘第一天
        stockData = np.vstack((stockData, [float(row[1])/20, float(row[4])/20, float(row[2])/20, float(row[3])/20, float(row[9])/100]))
        stockNote = np.vstack((stockNote, [row[0], row[5]]))
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
x0 = tf.placeholder(tf.float32, shape = [None, analyzeZone[0] * 5], name = "x0")
x1 = tf.placeholder(tf.float32, shape = [None, analyzeZone[1] * 5], name = "x1")
x2 = tf.placeholder(tf.float32, shape = [None, analyzeZone[2] * 5], name = "x2")
x3 = tf.placeholder(tf.float32, shape = [None, analyzeZone[3] * 5], name = "x3")
x4 = tf.placeholder(tf.float32, shape = [None, analyzeZone[4] * 5], name = "x4")
x5 = tf.placeholder(tf.float32, shape = [None, analyzeZone[5] * 5], name = "x5")
x6 = tf.placeholder(tf.float32, shape = [None, analyzeZone[6] * 5], name = "x6")
x7 = tf.placeholder(tf.float32, shape = [None, analyzeZone[7] * 5], name = "x7")

#level1
w00 = weight_variable([analyzeZone[0] * 5, analyzeL1[0]], name = "w00")
w10 = weight_variable([analyzeZone[1] * 5, analyzeL1[1]], name = "w10")
w20 = weight_variable([analyzeZone[2] * 5, analyzeL1[2]], name = "w20")
w30 = weight_variable([analyzeZone[3] * 5, analyzeL1[3]], name = "w30")
w40 = weight_variable([analyzeZone[4] * 5, analyzeL1[4]], name = "w40")
w50 = weight_variable([analyzeZone[5] * 5, analyzeL1[5]], name = "w50")
w60 = weight_variable([analyzeZone[6] * 5, analyzeL1[6]], name = "w60")
w70 = weight_variable([analyzeZone[7] * 5, analyzeL1[7]], name = "w70")
b00 = bias_variable([analyzeL1[0]], name = "b00")
b10 = bias_variable([analyzeL1[1]], name = "b10")
b20 = bias_variable([analyzeL1[2]], name = "b20")
b30 = bias_variable([analyzeL1[3]], name = "b30")
b40 = bias_variable([analyzeL1[4]], name = "b40")
b50 = bias_variable([analyzeL1[5]], name = "b50")
b60 = bias_variable([analyzeL1[6]], name = "b60")
b70 = bias_variable([analyzeL1[7]], name = "b70")
f00 = tf.nn.relu(tf.matmul(x0, w00) + b00, name = "f00")
f10 = tf.nn.relu(tf.matmul(x1, w10) + b10, name = "f10")
f20 = tf.nn.relu(tf.matmul(x2, w20) + b20, name = "f20")
f30 = tf.nn.relu(tf.matmul(x3, w30) + b30, name = "f30")
f40 = tf.nn.relu(tf.matmul(x4, w40) + b40, name = "f40")
f50 = tf.nn.relu(tf.matmul(x5, w50) + b50, name = "f50")
f60 = tf.nn.relu(tf.matmul(x6, w60) + b60, name = "f60")
f70 = tf.nn.relu(tf.matmul(x7, w70) + b70, name = "f70")

#level2
w01 = weight_variable([analyzeL1[0], analyzeL2[0]], name = "w01")
w11 = weight_variable([analyzeL1[1], analyzeL2[1]], name = "w11")
w21 = weight_variable([analyzeL1[2], analyzeL2[2]], name = "w21")
w31 = weight_variable([analyzeL1[3], analyzeL2[3]], name = "w31")
w41 = weight_variable([analyzeL1[4], analyzeL2[4]], name = "w41")
w51 = weight_variable([analyzeL1[5], analyzeL2[5]], name = "w51")
w61 = weight_variable([analyzeL1[6], analyzeL2[6]], name = "w61")
w71 = weight_variable([analyzeL1[7], analyzeL2[7]], name = "w71")
b01 = bias_variable([analyzeL2[0]], name = "b01")
b11 = bias_variable([analyzeL2[1]], name = "b11")
b21 = bias_variable([analyzeL2[2]], name = "b21")
b31 = bias_variable([analyzeL2[3]], name = "b31")
b41 = bias_variable([analyzeL2[4]], name = "b41")
b51 = bias_variable([analyzeL2[5]], name = "b51")
b61 = bias_variable([analyzeL2[6]], name = "b61")
b71 = bias_variable([analyzeL2[7]], name = "b71")
f01 = tf.nn.relu(tf.matmul(f00, w01) + b01, name = "f01")
f11 = tf.nn.relu(tf.matmul(f10, w11) + b11, name = "f11")
f21 = tf.nn.relu(tf.matmul(f20, w21) + b21, name = "f21")
f31 = tf.nn.relu(tf.matmul(f30, w31) + b31, name = "f31")
f41 = tf.nn.relu(tf.matmul(f40, w41) + b41, name = "f41")
f51 = tf.nn.relu(tf.matmul(f50, w51) + b51, name = "f51")
f61 = tf.nn.relu(tf.matmul(f60, w61) + b61, name = "f61")
f71 = tf.nn.relu(tf.matmul(f70, w71) + b71, name = "f71")

#middle out
preOut = tf.concat([f01,f11,f21,f31,f41,f51,f61,f71], axis = 1, name = "preOut")

#y = tf.placeholder(tf.float32, [4], name = "y") #开盘，收盘，最高，最低
#x = tf.placeholder(tf.float32, shape = [None, analyzeZone * 5], name = "x")
y_ = tf.placeholder(tf.float32, shape = [None, 1], name = "y_") #开盘，收盘，最高，最低
myStep = tf.placeholder(tf.float32, name = "myStep")

w1 = weight_variable([np.sum(analyzeL2), 50], name = "w1") #35
b1 = bias_variable([50], name = "b1")
f1 = tf.nn.relu(tf.matmul(preOut, w1) + b1, name = "f1")
#f1 = tf.matmul(x, w1) + b1

w4 = weight_variable([50, 10], name = "w4")
b4 = bias_variable([10], name = "b4")
f4 = tf.nn.relu(tf.matmul(f1, w4) + b4, name = "f4")
#f4 = tf.matmul(f1, w4) + b4

#第四层，测试和训练模式切换
keep_prob = tf.placeholder("float", name = "keep_prob")
h_fc1_drop = tf.nn.dropout(f4, keep_prob, name = "h_fc1_drop")

w9 = weight_variable([10, 1], "w9")
b9 = bias_variable([1], "b9")
y_conv = tf.nn.relu(tf.matmul(h_fc1_drop, w9) + b9)
#y_conv = tf.matmul(h_fc1_drop, w9) + b9

#精度统计
incReal = tf.add(tf.multiply(y_[0,0], 0.2), - 0.1);
incRun = tf.add(tf.multiply(y_conv[0,0], 0.2), -0.1);
accuracy =  tf.add(incRun, - incReal)

#cross_entropy = tf.reduce_sum((y_ - y_conv)*(y_ - y_conv))
#cross_entropy = accuracy * accuracy * 1000
cross_entropy = tf.multiply(tf.multiply(accuracy, accuracy), 10000)
train_step = tf.train.AdamOptimizer(myStep).minimize(cross_entropy, name = "train_step")
#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1), name = "correct_prediction")
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name = "accuracy")

#开始
from time import time

#初始化
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter("d://temp//tensorFlow//MnistL1",tf.get_default_graph())  
writer.close()

print("start")
plt.figure()

#循环训练
repeat = 0
i = 0
while i < stockDataLen - analyzeZone[0] - 2:
    if i == 0:
        plt.clf()
        plt.subplot(2,1,2)
        plt.plot([0, 0], [-10, 10],  color='g', linestyle='-', linewidth = 0.5)
        plt.plot([0, stockDataLen - analyzeZone[0]], [0,0],  color='g', linestyle='-', linewidth = 0.5)
        plt.subplot(2,1,1)
        plt.plot([0, 0], [-10, 10],  color='g', linestyle='-', linewidth = 0.5)
        plt.plot([0, stockDataLen - analyzeZone[0]], [0,0],  color='g', linestyle='-', linewidth = 0.5)
        print("=============循环%d=============="%(repeat))
    if i == stockDataLen - analyzeZone[0] - 2 - 1:
        repeat = repeat + 1
        if repeat > 10000: break
        i = 0
        continue
    i = i + 1
        
    batch = stockData[i * 5 : (i + analyzeZone[0]) * 5]
    resultTrain = [[
        ((stockData[(i + analyzeZone[0]) * 5 + 1]/
        stockData[(i + analyzeZone[0] - 1) * 5 + 1]) - 1 + 0.1) * 5
        ]]
    feedTest = {
        x0: np.reshape(batch[0: analyzeZone[0] * 5], (1, analyzeZone[0] * 5)),
        x1: np.reshape(batch[(analyzeZone[0] - analyzeZone[1]) * 5: analyzeZone[0] * 5], (1, analyzeZone[1] * 5)),
        x2: np.reshape(batch[(analyzeZone[0] - analyzeZone[2]) * 5: analyzeZone[0] * 5], (1, analyzeZone[2] * 5)),
        x3: np.reshape(batch[(analyzeZone[0] - analyzeZone[3]) * 5: analyzeZone[0] * 5], (1, analyzeZone[3] * 5)),
        x4: np.reshape(batch[(analyzeZone[0] - analyzeZone[4]) * 5: analyzeZone[0] * 5], (1, analyzeZone[4] * 5)),
        x5: np.reshape(batch[(analyzeZone[0] - analyzeZone[5]) * 5: analyzeZone[0] * 5], (1, analyzeZone[5] * 5)),
        x6: np.reshape(batch[(analyzeZone[0] - analyzeZone[6]) * 5: analyzeZone[0] * 5], (1, analyzeZone[6] * 5)),
        x7: np.reshape(batch[(analyzeZone[0] - analyzeZone[7]) * 5: analyzeZone[0] * 5], (1, analyzeZone[7] * 5)),
        y_: resultTrain,
        keep_prob: 1.0,
        myStep: 1e-4
        }
    feedRun = {
        x0: np.reshape(batch[0: analyzeZone[0] * 5], (1, analyzeZone[0] * 5)),
        x1: np.reshape(batch[(analyzeZone[0] - analyzeZone[1]) * 5: analyzeZone[0] * 5], (1, analyzeZone[1] * 5)),
        x2: np.reshape(batch[(analyzeZone[0] - analyzeZone[2]) * 5: analyzeZone[0] * 5], (1, analyzeZone[2] * 5)),
        x3: np.reshape(batch[(analyzeZone[0] - analyzeZone[3]) * 5: analyzeZone[0] * 5], (1, analyzeZone[3] * 5)),
        x4: np.reshape(batch[(analyzeZone[0] - analyzeZone[4]) * 5: analyzeZone[0] * 5], (1, analyzeZone[4] * 5)),
        x5: np.reshape(batch[(analyzeZone[0] - analyzeZone[5]) * 5: analyzeZone[0] * 5], (1, analyzeZone[5] * 5)),
        x6: np.reshape(batch[(analyzeZone[0] - analyzeZone[6]) * 5: analyzeZone[0] * 5], (1, analyzeZone[6] * 5)),
        x7: np.reshape(batch[(analyzeZone[0] - analyzeZone[7]) * 5: analyzeZone[0] * 5], (1, analyzeZone[7] * 5)),
        y_: resultTrain,
        keep_prob: 0.5,
        myStep: 1e-3
        }
    
    train_step.run(feed_dict = feedRun)
    if i%51 == 0:
        train_accuracy = accuracy.eval(feed_dict = feedTest)
        testIncRun = incRun.eval(feed_dict = feedTest)
        testIncReal = incReal.eval(feed_dict = feedTest)
        if testIncRun > 0:
            plt.subplot(2,1,2)
            plt.plot([i,i], [0, testIncReal * 100],  color='r', linestyle='-', linewidth = 2)
        
        #plt.plot(i, train_accuracy*100,  color='g', linestyle='-', linewidth = 2, marker='o')
        plt.subplot(2,1,1)
        plt.plot([i,i], [0, train_accuracy*100],  color='g', linestyle='-', linewidth = 2)
        result = y_conv.eval(feed_dict = feedTest)
        d1 = (stockData[(i + analyzeZone[0] - 1) * 5 + 0] +
            stockData[(i + analyzeZone[0] - 1) * 5 + 1]) / 2
        print("%d ==> percent: %f%%, 实际:%f%%,预测:%f%%, date=%s, 表格涨幅=%s"%(
            i,
            train_accuracy*100,
            testIncReal*100,
            testIncRun*100,
            stockNote[i + analyzeZone[0]][0], 
            stockNote[i + analyzeZone[0]][1] 
            ))
        plt.pause(0.001)
    #print(cross_entropy.eval(feed_dict={x: np.reshape(batch, (1, analyzeZone * 5)), y_: resultTrain, keep_prob: 1.0, myStep: 1e-4}))
    #print(resultTrain)
    #print(y_conv.eval(feed_dict={x: np.reshape(batch, (1, analyzeZone * 5)), y_: resultTrain, keep_prob: 1.0, myStep: 1e-4}))

#plt.show()












