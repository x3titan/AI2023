#coding=utf-8
#载入CSV文件范例
import tensorflow as tf
from array import array
import numpy as np


file_queue = tf.train.string_input_producer(["./data/inputTest.csv", ""])
reader = tf.TextLineReader(skip_header_lines=1)
#reader = tf.TextLineReader()
key, value = reader.read(file_queue)
defaults = [[0.], [0.], [0.], [0.], [0.], ['']]
field1,field2,field3,field4,field5,field6 = tf.decode_csv(value, defaults)

#case语句转换范例
#field6x = tf.case({
#    tf.equal(field6, tf.constant('Iris-setosa')): lambda: tf.constant(0),
#    tf.equal(field6, tf.constant('Iris-versicolor')): lambda: tf.constant(1),
#    tf.equal(field6, tf.constant('Iris-virginica')): lambda: tf.constant(2),
#    }, lambda: tf.constant(-1), exclusive=True)

data = tf.stack([field1, field2, field3, field4, field5])
#a = tf.Variable(field1)

print("=======通过tensorFlow的tf.TextLineReader载入csv===========")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
  
    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
  
#    for i in range(5):
#        print(sess.run([field1, data]))

    while not coord.should_stop():
        try:
            print(sess.run([field1, data]))
        except tf.errors.NotFoundError:
            print("load csv complete")
            break
        #except tf.errors.OutOfRangeError:
        #    print("load csv complete")
        #    break
            
    coord.request_stop()
    coord.join(threads)
    
    #print(sess.run([field1, data]))
    
print("=======通过csv功能包载入csv文件=======")
import csv
csv_reader = csv.reader(open("./stock/600019.csv", encoding="gb2312"))
i = 0
sOpen = np.array([])
sClose = np.array([])
#开盘，收盘，最高，最低，换手率
result2 =  np.empty(shape=[0, 5])
print (np.shape(result2))
for row in csv_reader:
    print(row)
    if i > 0: #跳过题头
        sOpen = np.append(sOpen, float(row[1]))
        sClose = np.append(sClose, float(row[4]))
        result2 = np.vstack((result2, [float(row[1]), float(row[4]), float(row[2]), float(row[3]), float(row[9])]))
    i = i + 1
    if i > 10:
        break
print(sOpen)
print(sClose)
print("sOpen长度 = %d"%(len(sOpen)))
result1 = np.transpose(np.vstack((sOpen, sClose)))
print(result1)
print(np.shape(result1))
#print(np.append(np.transpose(sOpen, 0), sClose.transpose()))
print(result2)
print(len(result2))


print("===============从列表写入csv文件===============")
import os
filename = "./temp/test.csv"
if not os.path.exists(filename):
    print("文件不存在，创建并初始化文件：" + filename)
    csvFile = open(filename,"w", newline="")
    writer = csv.writer(csvFile)
    writer.writerow(["字段A","字段B", "字段C"])
    csvFile.close()
csvFile = open(filename,"a", newline="")
writer = csv.writer(csvFile)
data = np.reshape(np.arange(30), (10,3))
for i in range(len(data)):
    writer.writerow(data[i])
csvFile.close()
print("写入文件完成：" + filename)

print(len(range(1,5)))    
    
    
       
