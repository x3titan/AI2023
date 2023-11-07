#coding=utf-8
import matplotlib.pyplot as plt
import time
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

plt.figure(1)
#plt.show(block = False)

for i in range(10):
    #time.sleep(0.3)
    print(i)
    plt.figure(1)
    plt.subplot(211)
    plt.plot([i,i], [0,i], color='b', linestyle='--', marker='o', label='y1 data')
    plt.subplot(212)
    plt.hist(i+1, i+1, normed=1, facecolor='g', alpha=0.75)
    plt.pause(0.001)
    
plt.show()

plt.clf()

plt.plot([[1,1],[2,2],[4,5]], color='b', linestyle='--', marker='o', label='y1 data')

plt.show()