#coding=utf-8
#高级一点的神经网络，包含边界分析，密集连接等等
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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

# 建立模型
x=tf.placeholder(tf.float32, [None, 28 * 28], name = "x")
y=tf.placeholder(tf.float32, [None, 10], name = "y")
y_ = tf.placeholder("float", shape=[None, 10], name = "y_")

#第一层5x5卷积 -> 32个输出
W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
b_conv1 = bias_variable([32], "b_conv1")

#图像转换
x_image = tf.reshape(x, [-1,28,28,1], name = "x_image")

#第一层输出
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1, name = "h_conv1")
h_pool1 = max_pool_2x2(h_conv1, "h_pool1")

#第二层5x5卷积 -> 64个输出
W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
b_conv2 = bias_variable([64], "b_conv2")

#第二层输出
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2, name = "h_conv2")
h_pool2 = max_pool_2x2(h_conv2, "h_pool2")

#第三层前端输入转化为7*7*64 -> 1024个输出
W_fc1 = weight_variable([7 * 7 * 64, 1024], name = "W_fc1")
b_fc1 = bias_variable([1024], name = "b_fc1")

#第三层输出
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name = "h_pool2_flat")
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name = "h_fc1")

#第四层，测试和训练模式切换
keep_prob = tf.placeholder("float", name = "keep_prob")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name = "h_fc1_drop")

#第五层，1024输入 -> 10输出
W_fc2 = weight_variable([1024, 10], "W_fc2")
b_fc2 = bias_variable([10], "b_fc2")

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name = "y_conv")

#精度统计
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv), name = "cross_entropy")
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name = "train_step")
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

writer = tf.summary.FileWriter("d://temp//tensorFlow//MnistL1",tf.get_default_graph())  
writer.close()

plt.figure()

#循环训练
for i in range(20000):
    batch = mnist.train.next_batch(50)
    #统计并输出准确率
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        wconv1 = sess.run(W_conv1, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        wconv1 = np.reshape(wconv1[1,1], [32])
        print(wconv1)
        plt.plot(wconv1,  color='g', linestyle='-', linewidth = 0.2)
        print("step %d, training accuracy %g"%(i,train_accuracy))
        plt.pause(0.001)
        
    if (i%1000==0):
        plt.clf()
        #plt.subplot(3,1,1)
        #plt.plot([i + 100, i + 1000], [0, 0],  color='g', linestyle='-')

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

#最终准确率测试
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

