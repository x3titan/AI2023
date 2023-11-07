#coding=utf-8
import tensorflow as tf;
import numpy as np;
import os;

v1 = tf.Variable(tf.zeros(10) , name="v1")

sess = tf.Session() 
sess.run(tf.global_variables_initializer())
print(sess.run(v1))  
saver = tf.train.Saver([v1])
saver.restore(sess, "./temp/v1.txt")
print("load complete")
print(sess.run(v1))
