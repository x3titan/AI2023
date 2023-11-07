#coding=utf-8
'''
Created on 2017-12-19
@author: tam
'''
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.font_manager as font
import numpy as np
import os
import csv
from test1.pub import tamPub1
from builtins import str
#from numpy import dtype
#from _overlapped import NULL
#import datetime

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

#v6改造想法：
#所有数据都采用变化形式，价格类指标全部以前一周期的开盘价格为准，去掉年月日标记


#——————————————————参数——————————————————————
#stockCode = "SH600858" #"SH600022"

#——————————————————内部实现——————————————————————
version = "V7"

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

#获取1日k线数据
def getDay1(stockCode):
    filename = "./stock/stockList/" + stockCode + ".csv"
    print("=======>> 开始载入日K线数据：", filename)
    csvFile = open(filename, encoding="gb2312")
    csv_reader = csv.reader(csvFile)
    #0日期，1序号，2345开盘收盘最高最低,6成交量,78910大盘开盘收盘最高最低,11成交量
    day1 =  np.empty(shape=[0, 12])
    i = 0
    for row in csv_reader:
        try:
            day1 = np.vstack((day1, [
                float(row[3]), 
                float(row[0]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
                float(row[8]),
                float(row[11]),
                float(row[12]),
                float(row[13]),
                float(row[14]),
                float(row[15])
                ]))
        except:
            print("发现日K线非法数据：", row)
            continue
        i = i + 1
    #填充day1为0数据
    for i in range(1, len(day1)):
        if day1[i][2 ] == 0: day1[i][2 ] = day1[i-1][2 ]
        if day1[i][3 ] == 0: day1[i][3 ] = day1[i-1][3 ]
        if day1[i][4 ] == 0: day1[i][4 ] = day1[i-1][4 ]
        if day1[i][5 ] == 0: day1[i][5 ] = day1[i-1][5 ]
        if day1[i][6 ] == 0: day1[i][6 ] = day1[i-1][6 ]
        if day1[i][7 ] == 0: day1[i][7 ] = day1[i-1][7 ]
        if day1[i][8 ] == 0: day1[i][8 ] = day1[i-1][8 ]
        if day1[i][9 ] == 0: day1[i][9 ] = day1[i-1][9 ]
        if day1[i][10] == 0: day1[i][10] = day1[i-1][10]
        if day1[i][11] == 0: day1[i][11] = day1[i-1][11]
    #产生结果集
    #0涨幅1日，1最高1日，2最低1日，3开盘1日
    #改造：2018-05-23
    #  0> 第1日最高和今日最高比较
    #  1> 第1日最低和今日最低比较
    #  x> 第2日最高和今日最高比较
    #  x> 第2日最低和今日最低比较
    #  x> 第3日最高和今日最高比较
    #  x> 第3日最低和今日最低比较
    day1Result = np.zeros(shape=(len(day1) - 10, 2))
    for i in range(0, len(day1Result)):
        day1Result[i][0] = (day1[i+1][4] / day1[i][3] - 1) * 100
        day1Result[i][1] = (day1[i+1][5] / day1[i][3] - 1) * 100
    #剔除除权数据
    for i in range(0, len(day1Result)):
        if day1Result[i][0] < -20: day1Result[i][0] = 0
        if day1Result[i][1] < -20: day1Result[i][1] = 0
    return day1, day1Result

#日线数据转换为增量数据
def getDay1Acc(day1):
    day1Acc = np.zeros(dtype = float, shape = (len(day1), np.shape(day1)[1] - 2))
    for i in range(0, len(day1)):
        prev = max(i - 1, 0)
        day1Acc[i][0] = day1[i][2] / day1[prev][2]
        day1Acc[i][1] = day1[i][3] / day1[prev][2]
        day1Acc[i][2] = day1[i][4] / day1[prev][2]
        day1Acc[i][3] = day1[i][5] / day1[prev][2]
        day1Acc[i][4] = day1[i][6] / day1[prev][6]
        day1Acc[i][5] = day1[i][7] / day1[prev][7]
        day1Acc[i][6] = day1[i][8] / day1[prev][7]
        day1Acc[i][7] = day1[i][9] / day1[prev][7]
        day1Acc[i][8] = day1[i][10] / day1[prev][7]
        day1Acc[i][9] = day1[i][11] / day1[prev][11]
    return day1Acc

def normDay1(stockCode, day1Acc, day1Result):
    #载入归一化数据
    normFilename1 = "./stock/min5/" + version + "Day1Nm" + stockCode + ".dat"
    normFilename2 = "./stock/min5/" + version + "Day1RNm" + stockCode + ".dat"
    if os.path.exists(normFilename1) and os.path.exists(normFilename2):
        print("载入日线归一化参数：" + normFilename1)
        day1Param = np.fromfile(normFilename1, dtype = float)
        day1Param = np.reshape(day1Param, (2, len(day1Param)//2))
        print("载入结果归一化参数：" + normFilename2)
        day1ResultParam = np.fromfile(normFilename2, dtype = float)
        day1ResultParam = np.reshape(day1ResultParam, (2, len(day1ResultParam)//2))
    else: 
        #计算归一化参数列表
        day1Param = np.zeros(shape=(2, np.shape(day1Acc)[1]))
        day1ResultParam = np.zeros(shape=(2, np.shape(day1Result)[1]))
        #统计
        day1Param[0] = np.min(day1Acc, axis=0)
        day1Param[1] = np.max(day1Acc, axis=0)
        day1ResultParam[0] = np.min(day1Result, axis=0)
        day1ResultParam[1] = np.max(day1Result, axis=0)
        day1Param.tofile(normFilename1)
        day1ResultParam.tofile(normFilename2)
        print("产生日线归一化列表完成，存盘：" + normFilename1)
        print("产生结果归一化列表完成，存盘：" + normFilename2)
    #归一化运算
    day1Norm = (day1Acc - day1Param[0]) / (day1Param[1] - day1Param[0])
    day1ResultNorm = (day1Result - day1ResultParam[0]) / (day1ResultParam[1] - day1ResultParam[0])
    print("归一化日线及结果数据完成")
    return day1Norm, day1ResultNorm, day1Param, day1ResultParam

def show4(stockCode, stockName, resultAi100, resultReal100, startSn):
    #涨幅1日，1最高1日，2最低1日，3开盘1日
    myfont = font.FontProperties(fname='C:/Windows/Fonts/msyh.ttf')  #微软雅黑字体
    showStart = startSn
    showEnd = showStart + len(resultAi100)
    plt.subplot(3, 2, 1)
    plt.tight_layout(pad = 1.1) #, h_pad, w_pad, rect)
    plt.cla()
    plt.plot([showStart, showStart],[-10,10], color='g')
    plt.plot([showStart, showEnd],[0,0], color='g')
    plt.plot([showStart, showEnd],[-10,-10], color="#00800040", linewidth = 1)
    plt.plot([showStart, showEnd],[10,10], color="#00800040", linewidth = 1)
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultReal100[:,0], (len(resultReal100))), 1, color="#b0c0ffff")
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultAi100[:,0], (len(resultAi100))), 0.4, color="#ff0000c0")
    plt.title(stockCode + stockName + " 1日涨幅", fontproperties=myfont)
    plt.subplot(3, 2, 3)
    plt.cla()
    plt.plot([showStart, showStart],[-10,10], color='g')
    plt.plot([showStart, showEnd],[0,0], color='g')
    plt.plot([showStart, showEnd],[-10,-10], color="#00800040", linewidth = 1)
    plt.plot([showStart, showEnd],[10,10], color="#00800040", linewidth = 1)
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultReal100[:,1], (len(resultReal100))), 1, color="#b0c0ffff")
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultAi100[:,1], (len(resultAi100))), 0.4, color="#ff0000c0")
    plt.title("1日最高", fontproperties=myfont)
    
    plt.subplot(3, 2, 2)
    plt.cla()
    plt.plot([showStart, showStart],[-10,10], color='g')
    plt.plot([showStart, showEnd],[0,0], color='g')
    plt.plot([showStart, showEnd],[-10,-10], color="#00800040", linewidth = 1)
    plt.plot([showStart, showEnd],[10,10], color="#00800040", linewidth = 1)
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultReal100[:,2], (len(resultReal100))), 1, color="#b0c0ffff")
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultAi100[:,2], (len(resultAi100))), 0.4, color="#ff0000c0")
    plt.title("1日最低", fontproperties=myfont)
    plt.subplot(3, 2, 4)
    plt.tight_layout(pad = 1.1) #, h_pad, w_pad, rect)
    plt.cla()
    plt.plot([showStart, showStart],[-10,10], color='g')
    plt.plot([showStart, showEnd],[0,0], color='g')
    plt.plot([showStart, showEnd],[-10,-10], color="#00800040", linewidth = 1)
    plt.plot([showStart, showEnd],[10,10], color="#00800040", linewidth = 1)
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultReal100[:,3], (len(resultReal100))), 1, color="#b0c0ffff")
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultAi100[:,3], (len(resultAi100))), 0.4, color="#ff0000c0")
    plt.title("1日开盘", fontproperties=myfont)
    
    plt.pause(0.1)
    
def show6(stockCode, stockName, resultAi100, resultReal100, startSn):
    show1(stockCode, stockName, resultAi100, resultReal100, startSn, 0, "第1日最高")
    show1(stockCode, stockName, resultAi100, resultReal100, startSn, 1, "第1日最低")
    #show1(stockCode, stockName, resultAi100, resultReal100, startSn, 2, "第2日最高")
    #show1(stockCode, stockName, resultAi100, resultReal100, startSn, 3, "第2日最低")
    #show1(stockCode, stockName, resultAi100, resultReal100, startSn, 4, "第3日最高")
    #show1(stockCode, stockName, resultAi100, resultReal100, startSn, 5, "第3日最低")
    plt.pause(0.1)
    
def show1(stockCode, stockName, resultAi100, resultReal100, startSn, dataIndex, displayName):
    myfont = font.FontProperties(fname='C:/Windows/Fonts/msyh.ttf')  #微软雅黑字体
    showStart = startSn
    showEnd = showStart + len(resultAi100)
    plt.subplot(2, 1, dataIndex + 1)
    plt.tight_layout(pad = 1.1) #, h_pad, w_pad, rect)
    plt.cla()
    plt.plot([showStart, showStart],[-10,10], color='g')
    plt.plot([showStart, showEnd],[0,0], color='g')
    plt.plot([showStart, showEnd],[-10,-10], color="#00800040", linewidth = 1)
    plt.plot([showStart, showEnd],[10,10], color="#00800040", linewidth = 1)
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultReal100[:,dataIndex], (len(resultReal100))), 1, color="#b0c0ffff")
    plt.bar(np.arange(showStart, showEnd), np.reshape(resultAi100[:,dataIndex], (len(resultAi100))), 0.4, color="#ff0000c0")
    plt.title(displayName, fontproperties=myfont)
    

def min5DecodeDate(value):
    v = value.split("/")
    if (len(v) != 3): return -1
    if (len(v[1]) == 1): v[1] = "0" + v[1]
    if (len(v[2]) == 1): v[2] = "0" + v[2]
    return (int(v[0] + v[1] + v[2]))

def min5DecodeTime(value):
    v = value.split(":")
    if (len(v) != 2): return -1
    v = np.array(v, dtype = int)
    if (v[0] < 12):
        return (v[0] - 9) * (60 // 5) - 35 // 5 + v[1] // 5
    else:
        return 2 * 60 // 5 + (v[0] - 13) * (60 // 5) + v[1] // 5 - 1
    
min5FindDate_buff = {}
def min5FindDate(min5Date, dateValue):
    if len(min5FindDate_buff) == 0:
        i = 0
        for a in min5Date:
            min5FindDate_buff[a] = i
            i = i + 1
    try:
        return min5FindDate_buff[dateValue]
    except:
        return -1;

#min5数据结构
# 0    1    2    3     4     5      6     7     8    9     10    /
#日期 开盘 最高  最低  收盘  总手  开盘 最高  最低  收盘  总手  开始循环，总共48*10个数据
def getMin5(stockCode, data):
    #stockCode = "SH600829"
    #载入已经合并好的数据
    min5Filename = "./stock/min5/min5" + stockCode + ".dat"
    if os.path.exists(min5Filename):
        min5 = np.fromfile(min5Filename, dtype = float)
        min5 = np.reshape(min5, (-1, 48 * 10 + 1))
        print("载入5分钟线数据完成：" + min5Filename)
        return min5

    #产生日期数据
    print("============开始合并5分钟线数据============")
    min5Date =  data[:,0:1]
    
    #构建空间
    min5 = np.hstack((min5Date, np.zeros(shape=(len(min5Date), 10 * 48))))
    min5Date = min5Date.reshape((-1))

    #读取所有的表并填充数据：当前股
    min5FindDate_buff = {} #初始化快速查找缓冲
    for year in range(2003, 2017 + 1):
        #产生文件名
        filename = "./stock/min5/" + stockCode + "_" + str(year) + ".csv"
        if not os.path.exists(filename):
            print("由于无数据，跳过文件：" + filename)
            continue
        print("正在合并5分钟线数据：" + filename)
        csvFile = open(filename, encoding="gb2312")
        csv_reader = csv.reader(csvFile)
        #文件数据格式：/0日期 /1时间 /2开盘 /3最高 /4最低 /5收盘 /6总手 /7交易额
        #1999/7/26    9:55    1590.54    1590.54    1587.74    1587.74    2673    2205280
        for row in csv_reader:
            if len(row) != 8: continue;
            rowDate = min5DecodeDate(row[0])
            rowTime = min5DecodeTime(row[1])
            if rowTime >= 48 or rowTime < 0:
                print("检测到非法时间数据：", row)
                continue;
            pos = min5FindDate(min5Date, rowDate)
            if pos < 0: continue
            min5[pos:pos+1, rowTime: rowTime + 5] = np.array(row[2:2+5], dtype = float)
        csvFile.close()
        
        #读取并填充数据：大盘
        if stockCode[0:2] == "SH":
            filename = "./stock/min5/" + "SH000001" + "_" + str(year) + ".csv"
            print("正在合并5分钟线上证指数：" + filename)
        else:
            filename = "./stock/min5/" + "SZ399001" + "_" + str(year) + ".csv"
            print("正在合并5分钟线深成指：" + filename)
        csvFile = open(filename, encoding="gb2312")
        csv_reader = csv.reader(csvFile)
        for row in csv_reader:
            if len(row) != 8: continue;
            rowDate = min5DecodeDate(row[0])
            rowTime = min5DecodeTime(row[1])
            if rowTime >=48:
                print("检测到大于收盘时间的数据：" , row)
                continue;
            pos = min5FindDate(min5Date, rowDate)
            if pos < 0: continue
            rowTime = rowTime * 10 + 5 + 1
            min5[pos:pos+1, rowTime: rowTime + 5] = np.array(row[2:2+5], dtype = float)
        csvFile.close()
    
    #对为零数据做特殊处理
    #寻找第一个非零数据
    lastOpen1 = 0
    lastOpen2 = 0
    lastVol1 = 0
    lastVol2 = 0
    for i in range(0, len(min5)):
        for j in range(0, 48):
            if lastOpen1 == 0:
                if min5[i][j*10+1] > 0: lastOpen1 =  min5[i][j*10+1]
            if lastOpen2 == 0:
                if min5[i][j*10+1+5] > 0: lastOpen2 =  min5[i][j*10+1+5]
            if lastVol1 == 0:
                if min5[i][j*10+1+4] > 0: lastVol1 =  min5[i][j*10+1+4]
            if lastVol2 == 0:
                if min5[i][j*10+1+4+5] > 0: lastVol2 =  min5[i][j*10+1+4+5]
        if lastOpen1 == 0: continue
        if lastOpen2 == 0: continue
        if lastVol1 == 0: continue
        if lastVol2 == 0: continue
        break
    #填充0数据
    for i in range(0, len(min5)):
        for j in range(0, 48):
            if min5[i][j*10+1] == 0: min5[i][j*10+1] = lastOpen1
            if min5[i][j*10+2] == 0: min5[i][j*10+2] = lastOpen1
            if min5[i][j*10+3] == 0: min5[i][j*10+3] = lastOpen1
            if min5[i][j*10+4] == 0: min5[i][j*10+4] = lastOpen1
            if min5[i][j*10+5] == 0: min5[i][j*10+5] = lastVol1
            if min5[i][j*10+6] == 0: min5[i][j*10+6] = lastOpen2
            if min5[i][j*10+7] == 0: min5[i][j*10+7] = lastOpen2
            if min5[i][j*10+8] == 0: min5[i][j*10+8] = lastOpen2
            if min5[i][j*10+9] == 0: min5[i][j*10+9] = lastOpen2
            if min5[i][j*10+10] == 0: min5[i][j*10+10] = lastVol2
            lastOpen1 = min5[i][j*10+1]
            lastVol1 = min5[i][j*10+5]
            lastOpen2 = min5[i][j*10+6]
            lastVol2 = min5[i][j*10+10]
    #转换为增量
    lastOpen1 = min5[0][1]
    lastVol1 = min5[0][5]
    lastOpen2 = min5[0][6]
    lastVol2 = min5[0][10]
    for i in range(0, len(min5)):
        for j in range(0, 48):
            currentOpen1 = min5[i][j*10+1]
            currenVol1 = min5[i][j*10+5]
            currenOpen2 = min5[i][j*10+6]
            currenVol2 = min5[i][j*10+10]
            min5[i][j*10+1] = min5[i][j*10+1] / lastOpen1
            min5[i][j*10+2] = min5[i][j*10+2] / lastOpen1
            min5[i][j*10+3] = min5[i][j*10+3] / lastOpen1
            min5[i][j*10+4] = min5[i][j*10+4] / lastOpen1
            min5[i][j*10+5] = min5[i][j*10+5] / lastVol1
            min5[i][j*10+6] = min5[i][j*10+6] / lastOpen2
            min5[i][j*10+7] = min5[i][j*10+7] / lastOpen2
            min5[i][j*10+8] = min5[i][j*10+8] / lastOpen2
            min5[i][j*10+9] = min5[i][j*10+9] / lastOpen2
            min5[i][j*10+10] = min5[i][j*10+10] / lastVol2
            lastOpen1 = currentOpen1
            lastVol1 = currenVol1
            lastOpen2 = currenOpen2
            lastVol2 = currenVol2
    print("=========" + stockCode + "5分钟线数据增量转换完成")

    #存盘
    print("=========" + stockCode + "5分钟线数据合成完毕，保存数据：" + min5Filename)
    min5.tofile(min5Filename)
    return min5

def normMin5(stockCode, min5):
    #stockCode = "SH600829"
    #载入归一化数据
    min5NormFilename = "./stock/min5/min5Nm" + stockCode + ".dat"
    if os.path.exists(min5NormFilename):
        min5Param = np.fromfile(min5NormFilename, dtype = float)
        min5Param = np.reshape(min5Param, (-1, 48 * 10 + 1))
        print("载入5分钟线归一化参数列表完成：" + min5NormFilename)
    else: 
        #计算归一化参数列表
        min5Param = np.zeros(shape=(2, 48 * 10 + 1))
        #日期不变换
        min5Param[1][0] = 1
        #生成个股值，个股成交，大盘值，大盘成交
        for i in range(0, 48):
            t = min5[:,i*10+1:i*10+1+4]
            if i == 0:
                v1 = t
            else:
                v1 = np.hstack((v1, t))
            t = min5[:,i*10+1+4:i*10+1+5]
            if i == 0:
                v2 = t
            else:
                v2 = np.hstack((v2, t))
            t = min5[:,i*10+1+5:i*10+1+5+4]
            if i == 0:
                v3 = t
            else:
                v3 = np.hstack((v3, t))
            t = min5[:,i*10+1+5+4:i*10+1+5+4+1]
            if i == 0:
                v4 = t
            else:
                v4 = np.hstack((v4, t))
        #统计
        min5Param[0:1, 1:1+4] = np.min(v1)
        min5Param[1:2, 1:1+4] = np.max(v1)
        min5Param[0:1, 1+4:1+4+1] = np.min(v2)
        min5Param[1:2, 1+4:1+4+1] = np.max(v2)
        min5Param[0:1, 1+5:1+5+4] = np.min(v3)
        min5Param[1:2, 1+5:1+5+4] = np.max(v3)
        min5Param[0:1, 1+5+4:1+5+4+1] = np.min(v4)
        min5Param[1:2, 1+5+4:1+5+4+1] = np.max(v4)
        #复制统计结果
        for i in range(1,48):
            min5Param[:,i*10+1:i*10+1+10] = min5Param[:,1:1+10]
        #存盘
        min5Param.tofile(min5NormFilename)
        print("产生5分钟归一化列表完成，存盘：" + min5NormFilename)
    #归一化运算
    min5 = (min5 - min5Param[0]) / (min5Param[1] - min5Param[0])
    print("归一化5分钟线数据完成")
    return min5, min5Param

def generateDnn(timeStep):
    #产生DNN网络
    Y = tf.placeholder(tf.float32, shape=[None, 2])
    X = tf.placeholder(tf.float32, shape=[None, timeStep * 10 + 48 * 10 * 3])
    #DnnOut, KeepProb = tamPub1.getMultDnn(X, [2400, 1380, 792, 475, 285, 171, 102, 61, 36, 22, 12, 7, 4], True)      #采用黄金分割方案
    DnnOut, KeepProb = tamPub1.getMultDnn(X, [2440, 4000, 4000, 2800, 1940, 1164, 690, 475, 285, 171, 102, 61, 36, 22, 12, 7, 4, 2], True)      #采用黄金分割方案
    
    #损失函数
    Loss = tf.reduce_sum(tf.square(DnnOut - Y)) * 30
    #Loss = Loss * Loss
    TrainOp = tf.train.AdamOptimizer(0.3).minimize(Loss) #1e-5
    ResultAi = DnnOut
    return X,Y,KeepProb, TrainOp,Loss,ResultAi

#mode 0:200日保留区间训练，1:测试模式，2:200日学习并输出结果，3:实盘状态
def aiStock1(mode, stockCode, stockName, day1, day1Norm, day1ResultNorm, day1Result, day1ResultParam, min5, refStockCode, testOffset = -25, testLength = 50):
    #参数设置
    batchSize = 10
    timeStep = 100   #取样天数
    testReservedRange = 200 #测试保留区大小
    trainSize = 1000
    #testOffset = -25   #测试开始位置，相对于测试保留区开始位置        
    #testLength = 50     #测试区间大小
    
    #初始化内部变量
    checkPointFilename = "./stock/save200/" + version + stockCode
    checkPointFilenameRef = "./stock/save200/" + version + refStockCode
    trainEnd = len(day1Norm) - batchSize - timeStep - 5 - testReservedRange
    testBegin = len(day1Norm) - testReservedRange + testOffset - timeStep
    testEnd = testBegin + testLength
    tf.reset_default_graph()
    #trainBegin = 10  #训练开始位置
    trainBegin = trainEnd - trainSize  #训练开始位置
    
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
    preTranningCount = 300
    for batch in range(1 if mode==2 else 1000):
        resultAi100 = np.empty(shape=[0, 2])
        resultReal100 = np.empty(shape=[0, 2])
        sumLoss = 0
        sumCount = 0
        if preTranningCount >= trainSize:
            preTranningCount = trainSize
        if mode==2:
            print("启动测试模式，测试区间：" + str(testBegin+timeStep-1) + "--" + str(str(testEnd+timeStep-1)))
            dataRange = range(testBegin, testEnd)
            keepProb = 1
            batchSize = 1
        else:
            if fullTraining:
                dataRange = range(trainBegin, trainEnd)
            else:
                dataRange = range(trainBegin, trainBegin + preTranningCount)
            keepProb = 0.5
        maxLoss100 = 0
        loss100 = 0
        maxLossIndexLast = maxLossIndex
        for i in dataRange:
            y = day1ResultNorm[i+timeStep-1:i+timeStep-1+batchSize,:]
            x = np.empty(shape=[batchSize, timeStep * np.shape(day1Norm)[1] + 48 * 10 * 3])
            for j in np.arange(i, i + batchSize):
                #日k线数据
                a = day1Norm[j:j+timeStep,:].reshape((-1))
                #最后三日5分钟线数据
                for k in np.arange(j+timeStep-3, j+timeStep):
                    a = np.hstack((a, min5[k:k+1,1:1+48*10].reshape((-1))))
                x[j-i] = a
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
            resultAi100 = np.vstack((resultAi100, resultAi[0] * (day1ResultParam[1] - day1ResultParam[0]) + day1ResultParam[0]))
            resultReal100 = np.vstack((resultReal100, y[0] * (day1ResultParam[1] - day1ResultParam[0]) + day1ResultParam[0]))
            
            #日志输出
            if i%100 == 0:
                if loss100>maxLoss100:
                    maxLossIndex = i
                    maxLoss100 = loss100
                loss100 = 0
                print("============", stockCode + stockName, "=========batch" + str(batch) + "====平均loss:", sumLoss/sumCount , "==========================")
                print(i, loss, resultAi[0:1,:6])
                print(i, loss, y[0:1,:4])
                if batch < 3:
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
                    resultAi100 = resultAi100[-100:,]; #仅显示100个数据
                    resultReal100 = resultReal100[-100:,];
                show6(
                    stockCode = stockCode, 
                    stockName = stockName, 
                    resultAi100 = resultAi100, 
                    resultReal100 = resultReal100, 
                    startSn = day1[i - len(resultAi100) + timeStep][1])
                resultAi100 = np.empty(shape=[0, 2])
                resultReal100 = np.empty(shape=[0, 2])
                
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
            if sumLoss/sumCount < 3.5 and not fullTraining:
                preTranningCount = preTranningCount + 100
                if preTranningCount > trainSize:
                    print("====> 训练达到标准正在保存神经网络：" + checkPointFilename)
                    saver = tf.train.Saver()
                    saver.save(sess, checkPointFilename)
                    return 0
                print("=================预处理达到标准，增加数据集至：", preTranningCount)
            if preTranningCount > trainSize / 3 * 2:
                print("====> 正在保存神经网络：" + checkPointFilename)
                saver = tf.train.Saver()
                saver.save(sess, checkPointFilename)
            if sumLoss/sumCount < 0.5 and not fullTraining:
                print("============================进入全数据训练模式=====================================")
                fullTraining = True
    sess.close()
    return 2

#保存200日模拟结果
def saveSim200(stockCode, stockName, date, data):
    filename = "./stock/result200/" + version + "Result200.csv"
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
def aiStockSim200(stockCode, stockName, day1, day1Norm, day1ResultNorm, day1Result, day1ResultParam, min5, refStockCode, testOffset = -25, testLength = 50):
    #参数设置
    batchSize = 10
    timeStep = 100   #取样天数
    trainBegin = 10  #训练开始位置
    testReservedRange = 200 #测试保留区大小
    testOffset = 0   #测试开始位置，相对于测试保留区开始位置        
    testLength = 150     #测试区间大小
    
    #初始化内部变量
    checkPointFilename = "./stock/save200/" + version + stockCode
    trainEnd = len(day1Norm) - batchSize - timeStep - 5 - 1 - testReservedRange
    testBegin = len(day1Norm) - testReservedRange + testOffset - timeStep
    testEnd = testBegin + testLength
    trainSize = 50
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

    resultAi100 = np.empty(shape=[0, 4])
    resultReal100 = np.empty(shape=[0, 4])
    resultDate = np.empty(shape=[0, 1])
    for day200 in range(testLength):
        sumLoss = 0
        sumCount = 0
        loss = 100
        batchSize = 10
        trainBeginLoop = testBegin - batchSize - trainSize - 6 + day200
        dataRange = range(trainBeginLoop, trainBeginLoop + trainSize)
        lossCount = 0
        while loss > 3:
            lossCount = lossCount + 1
            if (lossCount >= 30):
                print("!!! loss计算大于20次，跳过本次运算过程")
                break
            for i in dataRange:
                y = day1ResultNorm[i+timeStep-1:i+timeStep-1+batchSize,:]
                x = np.empty(shape=[batchSize, timeStep * np.shape(day1Norm)[1] + 48 * 10 * 3])
                for j in np.arange(i, i + batchSize):
                    #日k线数据
                    a = day1Norm[j:j+timeStep,:].reshape((-1))
                    #最后三日5分钟线数据
                    for k in np.arange(j+timeStep-3, j+timeStep):
                        a = np.hstack((a, min5[k:k+1,1:1+48*10].reshape((-1))))
                    x[j-i] = a
                _,loss, resultAi = sess.run([TrainOp, Loss, ResultAi], feed_dict = {
                    X:x,
                    Y:y,
                    KeepProb: 0.5
                })
                sumLoss = sumLoss + loss
                sumCount = sumCount + 1
            print(stockCode, stockName, "day:", day200, ", loss=", loss, ", lossCount=", lossCount)
        print("======>", stockCode, stockName, "day:", day200, ", loss=", loss, ", lossCount=", lossCount)
            
        #算一个结果
        batchSize = 1
        i = testBegin + day200
        y = day1ResultNorm[i+timeStep-1:i+timeStep-1+batchSize,:]
        x = np.empty(shape=[batchSize, timeStep * np.shape(day1Norm)[1] + 48 * 10 * 3])
        for j in np.arange(i, i + batchSize):
            #日k线数据
            a = day1Norm[j:j+timeStep,:].reshape((-1))
            #最后三日5分钟线数据
            for k in np.arange(j+timeStep-3, j+timeStep):
                a = np.hstack((a, min5[k:k+1,1:1+48*10].reshape((-1))))
            x[j-i] = a
        loss, resultAi = sess.run([Loss, ResultAi], feed_dict = {
            X:x,
            Y:y,
            KeepProb: 1
        })
        #收集统计数据
        resultAi100 = np.vstack((resultAi100, resultAi[0] * (day1ResultParam[1] - day1ResultParam[0]) + day1ResultParam[0]))
        resultReal100 = np.vstack((resultReal100, y[0] * (day1ResultParam[1] - day1ResultParam[0]) + day1ResultParam[0]))
        #resultDate = np.vstack((resultDate, stockData[i+timeStep-1:i+timeStep-1+batchSize, 33:33 + 1])) 

    sess.close()
        
    show4(
        stockCode = stockCode, 
        stockName = stockName, 
        resultAi100 = resultAi100, 
        resultReal100 = resultReal100, 
        startSn = day1[i - len(resultAi100) + timeStep][1])
    
    saveSim200(
        stockCode = stockCode, 
        stockName = stockName,
        date = resultDate,
        data = resultAi100)
    resultAi100 = np.empty(shape=[0, 4])
    resultReal100 = np.empty(shape=[0, 4])

#对股票列表进行训练
def trainStockList():
    #载入股票列表
    stockListTarget = loadTargetStockList(skipCount = 7) #10
    #refStockCode = stockListTarget[0][0]
    refStockCode = "SH600026"
    #循环计算每支股票
    #plt.figure(figsize=(5,3))
    plt.figure(figsize=(13,5.5))
    #特殊的测试股票
    #stockListTarget[0][0] = "SH600516"
    #stockListTarget[0][1] = "方大炭素"
    for i in range(len(stockListTarget)):
        if stockListTarget[i][0] == "SH601985": continue
        day1, day1Result = getDay1(stockListTarget[i][0])
        day1Acc = getDay1Acc(day1)
        day1Norm, day1ResultNorm, day1Param, day1ResultParam = normDay1(stockListTarget[i][0], day1Acc, day1Result)
        min5 = getMin5(stockCode = stockListTarget[i][0], data = day1)
        min5, min5Param = normMin5(stockListTarget[i][0], min5)
        resultCode = -1
        while resultCode < 0:
            resultCode = aiStock1(
                mode = 1,
                stockCode = stockListTarget[i][0],
                stockName = stockListTarget[i][1],
                day1 = day1,
                day1Norm = day1Norm,
                day1ResultNorm = day1ResultNorm,
                day1Result = day1Result,
                day1ResultParam = day1ResultParam,
                min5 = min5,
                refStockCode = refStockCode
                )
            if resultCode == 0:
                refStockCode = stockListTarget[i][0]

#对股票列表进行训练
def testStock(stockCode, testStart, testLength):
    #循环计算每支股票
    #plt.figure(figsize=(16,7))
    plt.figure(figsize=(13,5.5))
    day1, day1Result = getDay1(stockCode)
    day1Acc = getDay1Acc(day1)
    day1Norm, day1ResultNorm, day1Param, day1ResultParam = normDay1(stockCode, day1Acc, day1Result)
    min5 = getMin5(stockCode = stockCode, data = day1)
    min5, min5Param = normMin5(stockCode, min5)
    aiStock1(
        mode = 2,
        stockCode = stockCode,
        stockName = "",
        day1 = day1,
        day1Norm = day1Norm,
        day1ResultNorm = day1ResultNorm,
        day1Result = day1Result,
        day1ResultParam = day1ResultParam,
        min5 = min5,
        refStockCode = "",
        testOffset=testStart,
        testLength=testLength
        )


#200日模拟结果输出
def sim200():
    #载入股票列表
    stockListTarget = loadTargetStockList(skipCount = 0) #10
    refStockCode = stockListTarget[0][0]
    #循环计算每支股票
    plt.figure(figsize=(13,5.5))
    #特殊的测试股票
    #stockListTarget[0][0] = "SH600516"
    #stockListTarget[0][1] = "方大炭素"
    #stockListTarget[0][0] = "SH600050"
    #stockListTarget[0][1] = ""
    for i in range(len(stockListTarget)):
    #for i in range(1):
        day1, day1Result = getDay1(stockListTarget[i][0])
        day1Acc = getDay1Acc(day1)
        day1Norm, day1ResultNorm, day1Param, day1ResultParam = normDay1(stockListTarget[i][0], day1Acc, day1Result)
        min5 = getMin5(stockCode = stockListTarget[i][0], data = day1)
        min5, min5Param = normMin5(stockListTarget[i][0], min5)
        aiStockSim200(
            stockCode = stockListTarget[i][0],
            stockName = stockListTarget[i][1],
            day1 = day1,
            day1Norm = day1Norm,
            day1ResultNorm = day1ResultNorm,
            day1Result = day1Result,
            day1ResultParam = day1ResultParam,
            min5 = min5,
            refStockCode = refStockCode
            )


#主程序
trainStockList()
#testStock("SH600050", 0, 150)
#testStock("SH600516", -0, 100)
#testStock("SH600030", -100, 200)

#sim200()    

plt.show()




