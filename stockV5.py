#coding=utf-8
'''
Created on 2017-12-19
@author: tam
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as font
import numpy as np
import csv
import datetime
from test1.pub import tamPub1
from builtins import str
#from tensorflow.python.training.saver import _GetCheckpointFilename
#from numpy import shape
#from warnings import catch_warnings
#import string

#——————————————————csv原始数据——————————————————————
#0       1           2           3           4       5       6       7       8           9           10          11          12          13          14          15
#序号    股票代码    股票名称    交易日      开盘    收盘    最高    最低    成交量      大盘代码    大盘名字    开盘        收盘        最高        最低        成交量
#1       SH600022    N 济  钢    20040629    6.36    6.37    6.43    5.9     61968585    SH000001    上证指数    1384.643    1408.689    1409.376    1376.218    10654000
#2       SH600022    济南钢铁    20040630    6.3     6.14    6.35    6.07    23504612    SH000001    上证指数    1408.272    1399.162    1415.844    1394.676    8709699
#3       SH600022    济南钢铁    20040701    6.14    6.57    6.75    6.11    60029150    SH000001    上证指数    1398.198    1441.068    1445.102    1396.347    14580099
#4       SH600022    济南钢铁    20040702    6.48    6.58    6.6     6.42    27818599    SH000001    上证指数    1441.121    1441.19     1447.612    1428.074    11068765
#——————————————————读入转换后数据——————————————————————
#0.开盘变化
#1.收盘变化
#2.最高变化
#3.最低变化
#4.成交量变化
#下面为大盘数据
#5.开盘变化
#6.收盘变化
#7.最高变化
#8.最低变化
#9.成交量变化
#10.月份    #11.日期    #11.星期
#13-20空缺

#0.月份    #1.日期    #2.星期
#3,开盘价格    4,开盘变化
#5,收盘价格    6,收盘变化
#7,最高价格    8,最高变化
#9,最低价格    10,最低变化
#11.成交量      12,成交量变化
#下面为大盘数据
#13,开盘价格    12,开盘变化
#15,收盘价格    14,收盘变化
#17,最高价格    16,最高变化
#19,最低价格    18,最低变化
#21.成交量      22,成交量变化

#———————————————训练结果V5版本,全部是百分比涨幅,全部基于今日收盘价格———————————————————
#23.2日均    #24.3日均
#25.4日均    #26.5日均
#———————————————参考值———————————————————
#33.日期
#34.序号

#——————————————————参数——————————————————————
#stockCode = "SH600858" #"SH600022"

#——————————————————内部实现——————————————————————
#载入目标股票列表
def loadTargetStockList(skipCount):
    filename = "./stock/stockList/stockListTarget.csv"
    print("=======>> 开始载入目标股票列表：%s"%(filename))
    csvFile = open(filename, encoding="gb2312")
    reader = csv.reader(csvFile)
    result =  np.empty(shape=[0, 2]) #股票代码,股票名字
    i = 0
    for row in reader:
        if i >= skipCount + 1: #跳过题头
            result = np.vstack((result, row[1:3]))
    
        i = i + 1
    csvFile.close()
    print("=======>> 载入目标股票列表完成,count=", len(result))
    return result

#从csv载入数据
def loadStockData(stockCode):
    def dateDiff(v1, v2):
        try:
            v1 = datetime.datetime.strptime(v1,"%Y%m%d")
            v2 = datetime.datetime.strptime(v2,"%Y%m%d")
        except:
            return 1
        return (v1 - v2).days
    def toFloat(value):
        result = []
        value = value.reshape((-1))
        for item in value:
            result = np.hstack((result, float(item)))
        return result
    def getAcc(v1, v2):
        v1 = float(v1)
        v2 = float(v2)
        if v2 == 0:
            return 0.5
        return v1 / v2
    filename = "./stock/stockList/" + stockCode + ".csv"
    print("=======>> 开始载入股票数据：%s"%(filename))
    csvFile = open(filename, encoding="gb2312")
    csv_reader = csv.reader(csvFile)
    data =  np.empty(shape=[0, 35])
    row6 = np.zeros([6, 16])
    rowResult = np.zeros([35])
    i = 0
    for row in csv_reader:
        if i > 1: #跳过题头和开盘第一天
            if (row[4]=="0"): continue #屏蔽停牌区间
            row6 = row6[1:6,:];
            row6 = np.vstack((row6, row))
            #0.月份    #1.日期    #2.星期
            rowResult[0] = float(row6[5,3][4:6])
            rowResult[1] = float(row6[5,3][6:8])
            try:
                d1 = datetime.datetime.strptime(row6[5,3 ],"%Y%m%d")
            except:
                rowResult[2] = 0
            rowResult[2] = d1.weekday() 
            rowResult[3 ] = float(row6[5, 4 ])
            rowResult[4 ] = float(row6[5, 4 ]) - float(row6[4, 4 ])
            rowResult[5 ] = float(row6[5, 5 ])
            rowResult[6 ] = float(row6[5, 5 ]) - float(row6[4, 5 ])
            rowResult[7 ] = float(row6[5, 6 ])
            rowResult[8 ] = float(row6[5, 6 ]) - float(row6[4, 6 ])
            rowResult[9 ] = float(row6[5, 7 ])
            rowResult[10] = float(row6[5, 7 ]) - float(row6[4, 7 ])
            rowResult[11] = float(row6[5, 8 ])
            rowResult[12] = float(row6[5, 8 ]) - float(row6[4, 8 ])
            rowResult[13] = float(row6[5, 11])
            rowResult[14] = float(row6[5, 11]) - float(row6[4, 11])
            rowResult[15] = float(row6[5, 12])
            rowResult[16] = float(row6[5, 12]) - float(row6[4, 12])
            rowResult[17] = float(row6[5, 13])
            rowResult[18] = float(row6[5, 13]) - float(row6[4, 13])
            rowResult[19] = float(row6[5, 14])
            rowResult[20] = float(row6[5, 14]) - float(row6[4, 14])
            rowResult[21] = float(row6[5, 15])
            rowResult[22] = float(row6[5, 15]) - float(row6[4, 15])
            rowResult[33] = float(row6[5, 3 ])
            rowResult[34] = float(row6[5, 0 ])
            data = np.vstack((data, rowResult))
            last = np.shape(data)[0] - 1
            #23.2日均    #24.3日均
            #25.4日均    #26.5日均
            if (last >= 2):
                data[-3, 23] = (sum(toFloat(row6[-2:, 5:6])) / float(row6[-3, 5]) / 2 - 1) * 100
            if (last >= 3):
                data[-4, 24] = (sum(toFloat(row6[-3:, 5:6])) / float(row6[-4, 5]) / 3 - 1) * 100
            if (last >= 4):
                data[-5, 25] = (sum(toFloat(row6[-4:, 5:6])) / float(row6[-5, 5]) / 4 - 1) * 100
            if (last >= 5):
                data[-6, 26] = (sum(toFloat(row6[-5:, 5:6])) / float(row6[-6, 5]) / 5 - 1) * 100
        i = i + 1
    csvFile.close()
    print("=======>> 股票数据载入完毕，共载入%d条记录"%(len(data)))
    return data

#归一化数据，返回：数据结果，加成系数，乘算系数
def normalizationData(data):
    #data_mean = np.mean(data, axis = 0)
    #data_mean = np.zeros((np.shape(data)[1]))
    dataMul = np.std(data[10:210,:], axis = 0)
    dataMax = np.max(data[10:210,:], axis = 0)
    dataAdd = np.array([
        0,0,0,
        0, dataMul[4] * 2, #个股
        0, dataMul[6] * 2,
        0, dataMul[8] * 2,
        0, dataMul[10] * 2,
        0, dataMul[12] * 2,
        0, dataMul[14] * 2, #大盘
        0, dataMul[16] * 2,
        0, dataMul[18] * 2,
        0, dataMul[20] * 2,
        0, dataMul[22] * 2,
        100, 100, 100, #训练结果
        100, 100, 100,
        100, 100, 100,
        100,
        0, 0 #两个参考数据
        ])
    dataMul = dataMul * 4;
    dataMul[0] = 12;  
    dataMul[1] = 31;  
    dataMul[2] = 7;  
    dataMul[3] = dataMax[3] * 10;
    dataMul[5] = dataMax[5] * 10;
    dataMul[7] = dataMax[7] * 10;
    dataMul[9] = dataMax[9] * 10;
    dataMul[11] = dataMax[11] * 10;
    dataMul[13] = dataMax[13] * 10;
    dataMul[15] = dataMax[15] * 10;
    dataMul[17] = dataMax[17] * 10;
    dataMul[19] = dataMax[19] * 10;
    dataMul[21] = dataMax[21] * 10;
    dataMul[23] = dataAdd[23] * 2;
    dataMul[24] = dataAdd[24] * 2;
    dataMul[25] = dataAdd[25] * 2;
    dataMul[26] = dataAdd[26] * 2;
    dataMul[27] = dataAdd[27] * 2;
    dataMul[28] = dataAdd[28] * 2;
    dataMul[29] = dataAdd[29] * 2;
    dataMul[30] = dataAdd[30] * 2;
    dataMul[31] = dataAdd[31] * 2;
    dataMul[32] = dataAdd[32] * 2;
    dataMul[33] = 1;
    dataMul[34] = 1;
    result = (data + dataAdd)/dataMul
    #print("归一化数据完成，加成系数:", dataAdd);
    #print("归一化数据完成，乘算系数:", dataMul);
    #print (data[5:8,:])
    #print (result[5:8,:])
    print("归一化数据完成, len=%d"%(len(data)))
    print(result[20])
    return result, dataAdd, dataMul

def show4(stockCode, stockName, stockData, resultAi100, resultReal100, startSn):
    myfont = font.FontProperties(fname='C:/Windows/Fonts/msyh.ttf')  #微软雅黑字体
    showStart = startSn
    showEnd = showStart + len(resultAi100)
    plt.subplot(2, 2, 1)
    plt.tight_layout(pad = 1.1) #, h_pad, w_pad, rect)
    plt.cla()
    plt.plot([showStart, showStart],[-10,10], color='g')
    plt.plot([showStart, showEnd],[0,0], color='g')
    plt.plot([showStart, showEnd],[-10,-10], color="#00800040", linewidth = 1)
    plt.plot([showStart, showEnd],[10,10], color="#00800040", linewidth = 1)
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultReal100[:,0], (len(resultReal100))), 1, color="#b0c0ffff")
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultAi100[:,0], (len(resultAi100))), 0.4, color="#ff0000c0")
    plt.title(stockCode + stockName + " 2日平均", fontproperties=myfont)
    plt.subplot(2, 2, 3)
    plt.cla()
    plt.plot([showStart, showStart],[-10,10], color='g')
    plt.plot([showStart, showEnd],[0,0], color='g')
    plt.plot([showStart, showEnd],[-10,-10], color="#00800040", linewidth = 1)
    plt.plot([showStart, showEnd],[10,10], color="#00800040", linewidth = 1)
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultReal100[:,1], (len(resultReal100))), 1, color="#b0c0ffff")
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultAi100[:,1], (len(resultAi100))), 0.4, color="#ff0000c0")
    plt.title("3日平均", fontproperties=myfont)
    
    plt.subplot(2, 2, 2)
    plt.cla()
    plt.plot([showStart, showStart],[-10,10], color='g')
    plt.plot([showStart, showEnd],[0,0], color='g')
    plt.plot([showStart, showEnd],[-10,-10], color="#00800040", linewidth = 1)
    plt.plot([showStart, showEnd],[10,10], color="#00800040", linewidth = 1)
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultReal100[:,2], (len(resultReal100))), 1, color="#b0c0ffff")
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultAi100[:,2], (len(resultAi100))), 0.4, color="#ff0000c0")
    plt.title("4日平均", fontproperties=myfont)
    plt.subplot(2, 2, 4)
    plt.tight_layout(pad = 1.1) #, h_pad, w_pad, rect)
    plt.cla()
    plt.plot([showStart, showStart],[-10,10], color='g')
    plt.plot([showStart, showEnd],[0,0], color='g')
    plt.plot([showStart, showEnd],[-10,-10], color="#00800040", linewidth = 1)
    plt.plot([showStart, showEnd],[10,10], color="#00800040", linewidth = 1)
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultReal100[:,3], (len(resultReal100))), 1, color="#b0c0ffff")
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultAi100[:,3], (len(resultAi100))), 0.4, color="#ff0000c0")
    plt.title("5日平均", fontproperties=myfont)
    
    plt.pause(0.1)


def generateDnn(timeStep):
    #产生DNN网络
    Y = tf.placeholder(tf.float32, shape=[None, 4])
    X = tf.placeholder(tf.float32, shape=[None, timeStep * 23])
    #DnnOut, KeepProb = tamPub1.getMultDnn(X, [2400, 1380, 792, 475, 285, 171, 102, 61, 36, 22, 12, 7, 4], True)      #采用黄金分割方案
    DnnOut, KeepProb = tamPub1.getMultDnn(X, [1150, 690, 475, 285, 171, 102, 61, 36, 22, 12, 7, 4], True)      #采用黄金分割方案
    
    #损失函数
    Loss = tf.reduce_sum(tf.square(DnnOut - Y)) * 30
    Loss = Loss * Loss
    TrainOp = tf.train.AdamOptimizer(1e-4).minimize(Loss)
    ResultAi = DnnOut
    return X,Y,KeepProb, TrainOp,Loss,ResultAi

#mode 0:200日保留区间训练，1:测试模式，2:200日学习并输出结果，3:实盘状态
def aiStock1(mode, stockCode, stockName, stockData, stockDataRaw, stockDataAdd, stockDataMul, refStockCode, testOffset = -25, testLength = 50):
    #参数设置
    version = "V5"
    batchSize = 10
    timeStep = 50   #取样天数
    trainBegin = 10  #训练开始位置
    testReservedRange = 200 #测试保留区大小
    #testOffset = -25   #测试开始位置，相对于测试保留区开始位置        
    #testLength = 50     #测试区间大小
    
    #初始化内部变量
    checkPointFilename = "./stock/save200/" + version + stockCode
    checkPointFilenameRef = "./stock/save200/" + version + refStockCode
    trainEnd = len(stockData) - batchSize - timeStep - 5 - testReservedRange
    testBegin = len(stockData) - testReservedRange + testOffset - timeStep
    testEnd = testBegin + testLength
    tf.reset_default_graph()
    
    #检测数据的合法性
    if trainEnd < 100:
        print("=================此股票数据数量不足，不能进行ai分析==========================")
        return 3 
    

    #产生DNN网络
    X,Y,KeepProb, TrainOp,Loss,ResultAi = generateDnn(timeStep = timeStep)

    #写入模型
    #writer = tf.summary.FileWriter("d://temp//tensorFlow//MnistL1",tf.get_default_graph())  
    #writer.close()
   
    #开始训练
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    #读取历史训练结果
    saver = tf.train.Saver()
    try:
        saver.restore(sess, checkPointFilename)
        print("====> 读取上次训练保存点完成：" + checkPointFilename)
    except:
        print("无法获取上次训练保存点，载入其他推荐股票的神经网络数据" + checkPointFilename)
        try:
            saver.restore(sess, checkPointFilenameRef)
            print("====> 读取其他股票的神经网络信息完成：" + checkPointFilenameRef)
        except:
            print("无法读取其他股票的神经网络信息，重新开始训练" + checkPointFilename)
        
    fullTraining = False
    fullTrainingCount = 0
    maxLossIndex = 0
    maxLossIndexLast = 0
    for batch in range(1 if mode==2 else 1000):
        resultAi100 = np.empty(shape=[0, 4])
        resultReal100 = np.empty(shape=[0, 4])
        sumLoss = 0
        sumCount = 0
        if mode==2:
            print("启动测试模式，测试区间：" + str(testBegin+timeStep-1) + "--" + str(str(testEnd+timeStep-1)))
            dataRange = range(testBegin, testEnd)
            keepProb = 1
            batchSize = 1
        else:
            if fullTraining:
                dataRange = range(trainBegin, trainEnd)
            else:
                dataRange = range(trainBegin, trainBegin + 300)
            keepProb = 0.5
        maxLoss100 = 0
        loss100 = 0
        maxLossIndexLast = maxLossIndex
        for i in dataRange:
            y = stockData[i+timeStep-1:i+timeStep-1+batchSize, 23:23 + 4]
            x = np.empty(shape=[batchSize, timeStep * 23])
            for j in np.arange(i, i + batchSize):
                x[j-i] = stockData[j:j+timeStep, 0:23].reshape((-1))
            if (mode==2):
                loss, resultAi = sess.run([Loss, ResultAi], feed_dict = {
                    X:x,
                    Y:y,
                    KeepProb: keepProb
                })
            else:
                _,loss, resultAi = sess.run([TrainOp, Loss, ResultAi], feed_dict = {
                    X:x,
                    Y:y,
                    KeepProb: keepProb
                })
            sumLoss = sumLoss + loss
            sumCount = sumCount + 1
            loss100 = loss100 + loss

            #收集统计数据
            resultAi100 = np.vstack((resultAi100, resultAi[0] * stockDataMul[23:23+4] - stockDataAdd[23:23+4]))
            resultReal100 = np.vstack((resultReal100, y[0] * stockDataMul[23:23+4] - stockDataAdd[23:23+4]))
            
            #日志输出
            if i%100 == 0:
                if loss100>maxLoss100:
                    maxLossIndex = i
                    maxLoss100 = loss100
                loss100 = 0
                print("============", stockCode + stockName, "=========batch" + str(batch) + "====平均loss:", sumLoss/sumCount , "==========================")
                print(i, loss, resultAi[0:1,:6])
                print(i, loss, y[0:1,:4])
                for k in range(len(resultAi[0])):
                    if resultAi[0][k] == 0:
                        print("!!!==========", stockCode + stockName, "恒定为0，放弃分析")
                        sess.close()
                        return -1;

            #绘图输出
            draw = False
            if mode==2 and i == dataRange[-1]:
                draw = True
            elif mode==1 and i == maxLossIndexLast:
                draw = True
            if draw:
                if mode==1:
                    resultAi100 = resultAi100[-50:,]; #仅显示50个数据
                    resultReal100 = resultReal100[-50:,];
                show4(
                    stockCode = stockCode, 
                    stockName = stockName, 
                    stockData = stockData, 
                    resultAi100 = resultAi100, 
                    resultReal100 = resultReal100, 
                    startSn = stockData[i - len(resultAi100) + timeStep][34])
                resultAi100 = np.empty(shape=[0, 4])
                resultReal100 = np.empty(shape=[0, 4])
                
        #print("=======第", batch,"轮整体误差,3日:", diff[0]/len(dataRange),"%,5日:", diff[1]/len(dataRange), "%,10日:", diff[2]/len(dataRange),"%")
        print("============================批次处理完成", stockCode + stockName, "=====================================")
        if mode==1:
            print("平均loss=", sumLoss/sumCount)  #0.00749597366749
            if fullTraining:
                print("====> 正在保存神经网络：" + checkPointFilename)
                saver = tf.train.Saver()
                saver.save(sess, checkPointFilename)
                print("====> 保存神经网络完成：" + checkPointFilename)
                fullTrainingCount = fullTrainingCount + 1
                if fullTrainingCount >= 80:
                    print("!!!====> 竖子不可教也")
                    sess.close()
                    return 1
            if sumLoss/sumCount <= 0.003 and fullTraining:
                print("============================loss达到标准，训练圆满完成", stockCode + stockName, "=====================================")
                sess.close()
                return 0
            if sumLoss/sumCount < 25 and not fullTraining:
                print("============================进入全数据训练模式=====================================")
                fullTraining = True
    sess.close()
    return 2

#保存200日模拟结果
def saveSim200(stockCode, stockName, date, data):
    import os
    filename = "./stock/result200/V4Result200.csv"
    if not os.path.exists(filename):
        print("文件不存在，创建并初始化文件：" + filename)
        csvFile = open(filename,"w", newline="")
        writer = csv.writer(csvFile)
        writer.writerow(["日期", "股票代码", "股票名字",
            "明日最高", "明日最低", "明日收盘",
            "二日最高", "二日最低", "二日收盘",
            "三日最高", "三日最低", "三日收盘",
            "五日最高", "五日最低", "五日收盘"
             ])
        csvFile.close()
    csvFile = open(filename,"a", newline="")
    writer = csv.writer(csvFile)
    for i in range(len(data)):
        writer.writerow(np.hstack((date[i], stockCode, stockName ,data[i])))
    csvFile.close()
    print("写入文件完成：" + filename)



#200日真实模拟
def aiStockSim200(stockCode, stockName, stockData, stockDataRaw, stockDataAdd, stockDataMul):
    #参数设置
    version = "V5"
    batchSize = 10
    timeStep = 50   #取样天数
    trainBegin = 10  #训练开始位置
    testReservedRange = 200 #测试保留区大小
    testOffset = 0   #测试开始位置，相对于测试保留区开始位置        
    testLength = 200     #测试区间大小
    
    #初始化内部变量
    checkPointFilename = "./stock/save200/" + version + stockCode
    trainEnd = len(stockData) - batchSize - timeStep - 5 - 1 - testReservedRange
    testBegin = len(stockData) - testReservedRange + testOffset - timeStep
    testEnd = testBegin + testLength
    tf.reset_default_graph()
    
    #检测数据的合法性
    if trainEnd < 100:
        print("=================此股票数据数量不足，不能进行ai分析==========================")
        return 3 
   
    #产生DNN网络
    X,Y,KeepProb, TrainOp,Loss,ResultAi = generateDnn(timeStep = timeStep)

    #开始计算
    myfont = font.FontProperties(fname='C:/Windows/Fonts/msyh.ttf')  #微软雅黑字体
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #读取历史训练结果
    saver = tf.train.Saver()
    try:
        saver.restore(sess, checkPointFilename)
        print("====> 载入已经训练好的神经网络：" + checkPointFilename)
    except:
        print("无法载入已经训练好的神经网络:" + checkPointFilename)
        return

    resultAi100 = np.empty(shape=[0, 12])
    resultReal100 = np.empty(shape=[0, 12])
    resultDate = np.empty(shape=[0, 1])
    for day200 in range(200):
        sumLoss = 0
        sumCount = 0
        loss = 100
        batchSize = 10
        trainBeginLoop = testBegin - batchSize - 30 - 6 + day200
        dataRange = range(trainBeginLoop, trainBeginLoop + 30)
        lossCount = 0
        while loss > 1:
            lossCount = lossCount + 1
            if (lossCount >= 20):
                print("!!! loss计算大于20次，跳过本次运算过程")
                break
            for i in dataRange:
                y = stockData[i+timeStep-1:i+timeStep-1+batchSize, 21:21 + 12]
                x = np.empty(shape=[batchSize,timeStep * 11])
                for j in np.arange(i, i + batchSize):
                    x[j-i] = stockData[j:j+timeStep, 0:11].reshape((-1))
                _,loss, resultAi = sess.run([TrainOp, Loss, ResultAi], feed_dict = {
                    X:x,
                    Y:y,
                    KeepProb: 0.5
                })
                sumLoss = sumLoss + loss
                sumCount = sumCount + 1
        print(stockCode, stockName, "day:", day200, ", loss=", loss, ", lossCount=", lossCount)
            
        #算一个结果
        batchSize = 1
        i = testBegin + day200
        y = stockData[i+timeStep-1:i+timeStep-1+batchSize, 21:21 + 12]
        x = np.empty(shape=[batchSize,timeStep * 11])
        for j in np.arange(i, i + batchSize):
            x[j-i] = stockData[j:j+timeStep, 0:11].reshape((-1))
        loss, resultAi = sess.run([Loss, ResultAi], feed_dict = {
            X:x,
            Y:y,
            KeepProb: 1
        })
        #收集统计数据
        resultAi100 = np.vstack((resultAi100, resultAi[0] * stockDataMul[21:33] - stockDataAdd[21:33]))
        resultReal100 = np.vstack((resultReal100, y[0] * stockDataMul[21:33] - stockDataAdd[21:33]))
        resultDate = np.vstack((resultDate, stockData[i+timeStep-1:i+timeStep-1+batchSize, 33:33 + 1])) 

    sess.close()
        
    show4(
        stockCode = stockCode, 
        stockName = stockName, 
        stockData = stockData, 
        resultAi100 = resultAi100, 
        resultReal100 = resultReal100, 
        startSn = stockData[testBegin + timeStep][34])
    saveSim200(
        stockCode = stockCode, 
        stockName = stockName,
        date = resultDate,
        data = resultAi100)
    resultAi100 = np.empty(shape=[0, 12])
    resultReal100 = np.empty(shape=[0, 12])


#对股票列表进行训练
def trainStockList():
    #载入股票列表
    stockListTarget = loadTargetStockList(skipCount = 0) #10
    refStockCode = stockListTarget[45][0]
    #循环计算每支股票
    #plt.figure(figsize=(16,7))
    plt.figure(figsize=(13,5.5))
    for i in range(len(stockListTarget)):
        stockDataRaw = loadStockData(stockListTarget[i][0])
        stockData, stockDataAdd, stockDataMul = normalizationData(stockDataRaw)
        resultCode = -1
        while resultCode < 0:
            resultCode = aiStock1(
                mode = 1,
                stockCode = stockListTarget[i][0],
                stockName = stockListTarget[i][1],
                stockData = stockData,
                stockDataRaw = stockDataRaw,
                stockDataAdd = stockDataAdd,
                stockDataMul = stockDataMul,
                refStockCode = refStockCode
                )
            if resultCode == 0:
                refStockCode = stockListTarget[i][0]

#对股票列表进行训练
def testStock(stockCode, testStart, testLength):
    #循环计算每支股票
    #plt.figure(figsize=(16,7))
    plt.figure(figsize=(13,5.5))
    stockDataRaw = loadStockData(stockCode)
    stockData, stockDataAdd, stockDataMul = normalizationData(stockDataRaw)
    aiStock1(
        mode = 2,
        stockCode = stockCode,
        stockName = "",
        stockData = stockData,
        stockDataRaw = stockDataRaw,
        stockDataAdd = stockDataAdd,
        stockDataMul = stockDataMul,
        refStockCode = "",
        testOffset=testStart,
        testLength=testLength
        )


#200日模拟结果输出
def sim200():
    #载入股票列表
    stockListTarget = loadTargetStockList(skipCount = 4)
    refStockCode = stockListTarget[0][0]
    #循环计算每支股票
    plt.figure(figsize=(16,7))
    for i in range(len(stockListTarget)):
        stockDataRaw = loadStockData(stockListTarget[i][0])
        stockData, stockDataAdd, stockDataMul = normalizationData(stockDataRaw)
        aiStockSim200(
            stockCode = stockListTarget[i][0],
            stockName = stockListTarget[i][1],
            stockData = stockData,
            stockDataRaw = stockDataRaw,
            stockDataAdd = stockDataAdd,
            stockDataMul = stockDataMul
            )


#主程序
#trainStockList()
#testStock("SZ300051", 50, 100)
#testStock("SH600104", -25, 50)
sim200()    

plt.show()




