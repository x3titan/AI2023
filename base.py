#coding=utf-8
#一些基础的使用方法
import tensorflow as tf
from numpy import shape
import numpy as np
from test1.pub import tamPub1

a1 = tf.placeholder(tf.float32, shape = (4, 2))

#变量
a2 = tf.Variable(tf.zeros([4, 2]))
#update
a2 = tf.scatter_update(a2, [0, 1, 2, 3], [[1, 2], [3, 4], [7, 6], [7, 8]])
#a2a = tf.Variable(tf.zeros(shape=[2, 4]))
#a2a = a2a + 2
a2a = tf.Variable(tf.constant(2.0, shape = [2, 4])) #常量的使用
a3 = tf.Variable(tf.zeros([10]))

#线性乘法
#普通乘法如：a4 = a2 * a2a，a2及a2a必须大小一致，结果是把每个对应位置的元素相乘
a4 = tf.matmul(a2, a2a)
a4 = a4 +  tf.reduce_sum(a4, reduction_indices=[1])

#统计所有列的和
a6 = tf.reduce_sum(a2, reduction_indices=[1])

#求平均值
a7 = tf.reduce_mean(a6)

#取值
a8 = tf.gather_nd(a2, [1])

a9 = tf.Variable(tf.zeros([4, 2])) + 10;
a9_1 = tf.Variable(tf.zeros([2])) + 1
a9_2 = a9 + a9_1

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()  # local variables like epoch_num, batch_size

a11 = tf.truncated_normal([4, 5], stddev=0.1)


sess = tf.Session()
sess.run(init_op)
sess.run(local_init_op)

#print(sess.run(a1))
print(sess.run(a2))
print(sess.run(a3))
print("======矩阵乘法========")
print(sess.run(a4))
print("======reduce_sum,统计所有行的和========")
print(sess.run(a6))
print("======reduce_mean,求平均值========")
print(sess.run(a7))
print("======a8 gather_nd 数组取值========")
print(sess.run(a8))

print("======a9========")
print(sess.run(a9))
print(sess.run(a9_1))
print(sess.run(a9_2))

print("======a2最大值的位置========")
print(sess.run(tf.argmax(a2,1)))

print("======s0:估计是以0为中心产生正态分布的随机数========")
s0 = sess.run(tf.truncated_normal([4,5], stddev=0.1))
#s0 = np.reshape(np.arange(25) + 0.0, [5,5])
print(s0)

print("======取s0的第1行========")
print(s0[1])
print("======取s0的第2列========")
print(s0[:,2])
print("======取s0的第一行第一列元素========")
print(s0[1,1])
print("======冒号代表范围，逗号分隔维度========")
print(s0[1:3,:2,np.newaxis])
print(np.newaxis)
print(s0.tolist())

print("======s0 softmax 计算一行中各个量的百分比，越大的数字百分比越高，百分比总和为1========")
print(sess.run(tf.nn.softmax(s0)))

print("======reshape,矩阵重组，-1代表可变长度========")
print(sess.run(tf.reshape(tf.truncated_normal([2, 8], stddev=0.1), [-1, 2])))

print("======convert2d========")
#第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，具有[batch, in_height, in_width, in_channels]这样的shape，
#具体含义是[训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]，注意这是一个4维的Tensor，要求类型为float32和float64其中之一
#第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
#具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同，有一个地方需要注意，第三维in_channels，就是参数input的第四维
#第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4

#太卡，暂时屏蔽一下
#c2d = tf.nn.conv2d(tf.truncated_normal([4, 4, 4, 1], stddev=0.1), tf.truncated_normal([2, 2, 1, 1], stddev=0.1), strides=[1, 1, 1, 1], padding='SAME')
#print(sess.run(c2d))
#print(c2d)

print("======relu，强制将小于零的数值设置为0========")
print(sess.run(tf.nn.relu(tf.truncated_normal([4, 5], stddev=0.1))))

print("======矩阵堆叠========")
c3 = np.reshape(np.arange(20), (4,5))
print(c3)
print(c3*[2,2,3,3,3])
c3 = np.vstack((c3, np.reshape(np.arange(10), (2,5))))
print(c3)
c3 = np.hstack((c3, np.reshape(np.arange(18), (6,3))))
print(c3)
print(sess.run(tf.concat([np.reshape(np.arange(15), (3,5)), np.reshape(np.arange(15), (3,5))], axis = 1)))

print("======求和========")
c3 = np.reshape(np.arange(20), (4,5))
print(c3)
print(np.sum(c3))
print(np.sum(c3, axis = 0))
print(np.sum(c3, axis = 1))

print("=======np.mean平均数=======")
print(np.mean(c3, axis = 0))

print("=======np.std标准差，均方差=======")
print(np.std(c3, axis = 0))

#print("a11:", sess.run(a11))
#print("tf.argmax(a11):", sess.run(a11, tf.argmax(a11)))
#tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1), name = "correct_prediction")
print("=======求均值=======")
c3 = np.arange(10)
filter = np.empty((3))*0+1
print(c3)
print(np.convolve(c3,filter,"full"))
print(np.convolve(c3,filter,"same"))
print(np.convolve(c3,filter,"valid"))
print("5日均线:", tamPub1.getMA(c3, 5))
print("归一化：", tamPub1.normalizeData(c3))

c3 = np.reshape(c3, (2,5))
c3[1] = [12314,344,1,1,1]
print(c3)
print("最大值：", np.max(c3, axis=0))


c3 = np.hstack((range(0,30,5)))
print(c3)












