#coding=utf-8
#从meta文件中读取数据
import tensorflow as tf;
import numpy as np;

sess = tf.Session() 
new_saver = tf.train.import_meta_graph("./temp/v1.ckpt.meta")
new_saver.restore(sess, "./temp/v1.ckpt")
  # tf.get_collection() 返回一个list. 但是这里只要第一个参数即可
v2 = tf.get_collection('pred_network')[0]

sess.run(tf.global_variables_initializer())

print(sess.run(v2))
