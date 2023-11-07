#coding=utf-8
#数据存入meta文件
import tensorflow as tf;
import numpy as np;
import os;

#train_X = np.linspace(-1, 1, 100);
#train_Y = 2 * train_X + np.random.rand(*train_X.shape) * 0.33 + 10

# 变量的初始化及输出

v1 = tf.Variable(np.linspace(-100, 100, 11), name="v1")
v2 = tf.Variable(tf.zeros([10]))

sess = tf.Session() 
sess.run(tf.global_variables_initializer())
print(sess.run(v1))  
#print(sess.run(v2))  
tf.add_to_collection('pred_network',  v1)
saver = tf.train.Saver([v1])
saver.save(sess, "./temp/v1.ckpt")
print("save complete")

#v1 =  tf.add(v1, 100)
#print(sess.run(v1))

v1 = tf.Variable(tf.zeros(10) , name="v1")
sess.run(tf.global_variables_initializer())
print(sess.run(v1))

#sess.run(v1)
#sav1 = tf.train.Saver([v1])
saver.restore(sess, "./temp/v1.ckpt")
print("load complete")
sess.run(tf.global_variables_initializer())
print(sess.run(v1))



# Add ops to save and restore only 'v2' using the name "my_v2"
#saver = tf.train.Saver({"my_v2": v2})
# Use the saver object normally after that.



