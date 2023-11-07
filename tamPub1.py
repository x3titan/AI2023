#coding=utf-8
import tensorflow as tf
import numpy as np

#随机正态分布
def weight_variable(shape, name = "variable"):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name = name)

#常量
def bias_variable(shape, name = "variable"):
    initial = tf.constant(0.6, shape=shape)
    return tf.Variable(initial, name = name)

#卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#右对齐均线
def getMA(x, duration):
    filter = np.empty((duration))*0+1
    return np.convolve(x,filter,"full")[0:-(duration-1)]

def normalizeData(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std, mean, std

#池化
def max_pool_2x2(x, name = "variable"):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name = name)

def getDnn(input, countIn, countOut):
    w = weight_variable([countIn, countOut], name = "w")
    b = bias_variable([countOut], name = "b")
    f = tf.nn.relu(tf.matmul(input, w) + b, name = "f")
    return f

def getMultDnn(input, nodeCount = [100,100,100], needDropout = True):
    layer = input
    keep_prob = tf.placeholder("float", name = "keep_prob")
    for i in range(len(nodeCount) - 1):
        if i==len(nodeCount)-1 and needDropout:
            layer = tf.nn.dropout(layer, keep_prob, name = "layerDropout")
        layer = getDnn(layer, nodeCount[i], nodeCount[i + 1])
    return layer, keep_prob

