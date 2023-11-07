#coding=utf-8
#采用LSTM神经网络预测股票数据
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv
from test1.pub import tamPub1
from tensorflow.python.training.saver import _GetCheckpointFilename
from numpy import shape

#定义常量
rnn_unit=800       #hidden layer units
input_size=5
output_size=3
lr=0.0006         #学习率
#——————————————————导入数据——————————————————————
#dataset_2.csv数据取样
#index_code,date,open,close,low,high,volume,money,change,label
#sh000001,1990/12/20,104.3,104.39,99.98,104.39,197000,85000,0.044108822,109.13
#sh000001,1990/12/21,109.07,109.13,103.73,109.13,28000,16100,0.045406648,114.55
#sh000001,1990/12/24,113.57,114.55,109.13,114.55,32000,31100,0.049665537,120.25
#sh000001,1990/12/25,120.09,120.25,114.55,120.25,15000,6500,0.04975993,125.27
#f=open('dataset_2.csv') 
#df=pd.read_csv(f)     #读入股票数据
#data=df.iloc[:,2:10].values  #取第3-10列

#600019.csv数据取样
#时间    1开盘    2最高    3最低    4收盘    5涨幅    6振幅    7总手    8金额    9换手%
#10成交次数    11明日收盘    12明日涨幅    13明3日涨幅    14明5日涨幅    15明10日涨幅
#2000-12-12,二    4.96    7.36    4.88    6.09    --    --    312,239,700    0    69.39    0    5.71
#2000-12-13,三    5.81    5.94    5.69    5.71    -6.24%    4.11%    95,947,400    0    21.32    0    5.58
filename = "./stock/600019.csv"
print("=======>> 开始载入股票数据：%s"%(filename))
csv_reader = csv.reader(open(filename, encoding="gb2312"))
#0开盘，1收盘，2最高，3最低，4换手率，5明日收盘，6明日涨幅，7明3日涨幅,8明5日涨幅,9明10日涨幅
data =  np.empty(shape=[0, 10])
dataResult = np.empty(shape=[0, 3])
i = 0
for row in csv_reader:
    if i > 1: #跳过题头和开盘第一天
        v = float(row[12])
        if v>=0.01:
            v = [0,0,1]
        elif v<=-0.01:
            v = [1,0,0]
        else:
            v = [0,1,0]
        data = np.vstack((data, [float(row[1]), float(row[4]), float(row[3]), float(row[2]), float(row[9]), float(row[11]), float(row[12]), float(row[13]), float(row[14]), float(row[15])]))
        dataResult = np.vstack((dataResult,v))
    i = i + 1
print("=======>> 股票数据载入完毕，共载入%d条记录"%(len(data)))

#获取训练集
def get_train_data(batch_size=1,time_step=20,train_begin=0,train_end=2800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    data_result=dataResult[train_begin:train_end]
    train_x,train_y=[],[]   #训练集 
    for i in range(len(data_train)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=data_train[i:i+time_step,:5]
       y=data_result[i:i+time_step]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(data_train)-time_step))
    return batch_index,train_x,train_y,data_train

#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
    'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
    'out':tf.Variable(tf.random_normal([rnn_unit,1]))
}
biases={
    'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
    'out':tf.Variable(tf.constant(0.1,shape=[1,]))
}

#随机正态分布
def weight_variable(shape, name = "variable"):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name = name)

#常量
def bias_variable(shape, name = "variable"):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

#——————————————————定义神经网络变量——————————————————
def lstm(X):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']  
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

# 定义RNN  
def neural_network(X, model='lstm', batch_size=1, rnn_size=20, num_layers=1):  
    if model == 'rnn':  
        cell_fun = tf.nn.rnn_cell.BasicRNNCell  
    elif model == 'gru':  
        cell_fun = tf.nn.rnn_cell.GRUCell  
    elif model == 'lstm':  
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell  

    cell = cell_fun(rnn_size, state_is_tuple=True)  
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)  

    initial_state = cell.zero_state(batch_size, tf.float32)  

    outputs, last_state = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state, scope='rnnlm')  
    output = tf.reshape(outputs,[-1, rnn_size])  

    softmax_w = weight_variable([rnn_size, 3], name = "softmax_w")
    softmax_b = bias_variable([3], name = "softmax_b")
    logits = tf.nn.relu(tf.matmul(output, softmax_w) + softmax_b)  
    probs = tf.nn.softmax(logits)  
    return probs

def convertResult(result):
    b1 = "↑"
    if result<0: b1 = "↓"
    if result==0: b1 = "="
    return b1

print("————————————————计算并预测————————————————————")
def tam_lstm(batch_size=80,time_step=30,train_begin=10,train_end=2800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size]) 
    #pred,finalState=lstm(X)
    probs = neural_network(X)
    print("probs:", probs)
    
    #损失函数
    loss = -tf.reduce_sum(Y[0][-1] * tf.log(tf.maximum(probs[-1], 0.01)), name = "cross_entropy")
    train_op=tf.train.AdamOptimizer(1e-4).minimize(loss)
    result = tf.argmax(Y[0], axis=1)[-1] - 1
    resultAi = tf.argmax(probs, axis=1)[-1] - 1
    resultEqual = tf.equal(result, resultAi)
    resultEqual = tf.reduce_mean(tf.cast(resultEqual, "float"), name = "accuracy")
    #准确率
    #p1 = tf.reshape(pred,[-1])[time_step-1]/5-0.1
    #p2 = tf.reshape(Y, [-1])[time_step-1]/5-0.1
    acc = tf.Variable(0.0) # p1-p2
    
    #载入数据
    batch_index,train_x,train_y,data_src=get_train_data(batch_size,time_step,train_begin,train_end)
    print("trainx:", np.shape(train_x))
    print("trainy:", np.shape(train_y))
    
    #开始训练
    plt.figure()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(train_end-train_begin-time_step):
        _,_loss,_acc,_result,_resultAi,_probs=sess.run([train_op,loss,acc,result,resultAi,probs],feed_dict={
            X:train_x[i:i+1],
            Y:train_y[i:i+1]
        })
        print(_result, _resultAi, _loss)
        print(_probs[-1])
        #print(np.shape(_finalState))
        #0开盘，1收盘，2最高，3最低，4换手率，5明日收盘
        #print("pred_",np.shape(_pred))
        #print("train_y",np.shape(train_y))
        #print("train_y[i:i+1]", np.shape(train_y[i:i+1]))
        print(
            "序号:",i+train_begin+time_step+2,
            ", 涨幅",convertResult(_result),(data_src[i+time_step-1][5]/5-0.1)*100,"%",
            ",预测",convertResult(_resultAi),
            ",实际:",data_src[i+time_step-1][0]*100,"元",
            ",预测:","元",
            ",差值:",_acc*100,"%"
        )
        if (i%50==0):
            plt.subplot(3,1,1)
            plt.cla()
            plt.plot([i, i+50], [0,0], color='g',  linewidth = 0.6)
            plt.plot([i, i], [-10, 10], color='g',  linewidth = 0.6)
            plt.subplot(3,1,2)
            plt.cla()
            plt.plot([i, i+50], [0,0], color='g',  linewidth = 0.6)
            plt.plot([i, i], [-10, 10], color='g',  linewidth = 0.6)
            plt.subplot(3,1,3)
            plt.cla()
            plt.plot([i, i+50], [0,0], color='g',  linewidth = 0.6)
            plt.plot([i, i], [-10, 10], color='g',  linewidth = 0.6)
        v1 = _resultAi*10
        v2 = (data_src[i+time_step-1][5]/5-0.1)*100
        plt.subplot(3,1,1)
        plt.plot([i,i], [0,_acc*100],  color='b', linewidth = 4.0)
        plt.subplot(3,1,2)
        if (v1*v2>=0):
            plt.plot([i,i], [0,v1],  color='r', linewidth = 4.0)
        else:
            plt.plot([i,i], [0,v1],  color='g', linewidth = 4.0)
        plt.subplot(3,1,3)
        if (v2>=0):
            plt.plot([i,i], [0,v2],  color='r', linewidth = 4.0)
        else:
            plt.plot([i,i], [0,v2],  color='g', linewidth = 4.0)
        plt.pause(0.001)

def tam_dnn(batch_size=80,time_step=300,train_begin=10,train_end=2800):
    #产生dnn网络
    X=tf.placeholder(tf.float32, shape=[None,time_step*input_size])
    Y=tf.placeholder(tf.float32, shape=[None,output_size]) 
    dnn, dropOut = tamPub1.getMultDnn(X, [time_step*input_size, 800, 400, 200, 50, 3], True)
    #resultAi = dnn
    
    #损失函数
    dnnSoftmax = tf.nn.softmax(dnn)
    loss = -tf.reduce_sum(Y * tf.log(tf.maximum(dnnSoftmax, 0.0001)), name = "loss")
    train_op=tf.train.AdamOptimizer(1e-6).minimize(loss)
    resultAi = dnnSoftmax
    #resultAi = tf.argmax(dnn,1)

    #writer = tf.summary.FileWriter("d://temp//tensorFlow//MnistL1",tf.get_default_graph())  
    #writer.close()

    #准备数据
    data = data * [0.01, 0.01, 0.01, 0.01, 0.01, 5]
    data = data + [0, 0, 0, 0, 0, 0.5]

    #开始训练
    plt.figure()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(2800):
        #载入数据
        trainX = np.reshape(data[i:i+time_step,:input_size], (1,time_step*input_size))
        trainY = np.reshape(dataResult[i+time_step-1:i+time_step], (1,3))
        _,_loss,_resultAi=sess.run([train_op,loss,resultAi],feed_dict={
            X:trainX,
            Y:trainY,
            dropOut: 0.5
        })
        print(i,_loss,trainY,_resultAi)

def tam_dnn2(batch_size=80,timeStep=10,train_begin=10,train_end=2800):
    #产生dnn网络
    Y = tf.placeholder(tf.float32, shape=[None, 3]) 
    #开盘价变化
    X1 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn1, _ = tamPub1.getMultDnn(X1, [timeStep, 100, 30], False)
    #收盘价变化
    X2 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn2, _ = tamPub1.getMultDnn(X1, [timeStep, 100, 30], False)
    #最高价变化
    X3 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn3, _ = tamPub1.getMultDnn(X1, [timeStep, 100, 30], False)
    #最低价变化
    X4 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn4, _ = tamPub1.getMultDnn(X1, [timeStep, 100, 30], False)
    #换手率变化
    X5 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn5, _ = tamPub1.getMultDnn(X1, [timeStep, 100, 30], False)
    #综合输出
    XOut = tf.concat([Dnn1,Dnn2,Dnn3,Dnn4,Dnn5], axis = 1, name = "XOut")
    DnnOut, KeepProb = tamPub1.getMultDnn(XOut, [150, 50, 3], True)
    
    #损失函数
    DnnSoftmax = tf.nn.softmax(DnnOut)
    Loss = -tf.reduce_sum(Y * tf.log(tf.maximum(DnnSoftmax, 0.0001)), name = "loss")
    Loss = Loss*Loss
    TrainOp = tf.train.AdamOptimizer(1e-4).minimize(Loss)
    #ResultAi = DnnSoftmax
    ResultAi = tf.argmax(DnnSoftmax,1)-1

    #写入模型
    #writer = tf.summary.FileWriter("d://temp//tensorFlow//MnistL1",tf.get_default_graph())  
    #writer.close()

    #准备数据
    global data
    global dataResult
    x1 = data[1:,0] - data[0:-1,0]
    x2 = data[1:,1] - data[0:-1,1]
    x3 = data[1:,2] - data[0:-1,2]
    x4 = data[1:,3] - data[0:-1,3]
    x5 = data[1:,4] - data[0:-1,4]
    x5 = x5 / 5 #换手率

    #开始训练
    plt.figure()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(2800):
        y = np.reshape(dataResult[i+timeStep-1:i+timeStep], (1,3))
        _,loss, resultAi = sess.run([TrainOp, Loss, ResultAi], feed_dict = {
            X1:np.reshape(x1[i:i+timeStep], (1,timeStep)),
            X2:np.reshape(x2[i:i+timeStep], (1,timeStep)),
            X3:np.reshape(x3[i:i+timeStep], (1,timeStep)),
            X4:np.reshape(x4[i:i+timeStep], (1,timeStep)),
            X5:np.reshape(x5[i:i+timeStep], (1,timeStep)),
            Y:y,
            KeepProb: 0.5
        })
        print(i,loss,y,resultAi)

#2017/10/31验证比较成功的模型,"./stock/stockLSTM2v1.ckpt"是训练4小时的数据
def tam_dnn3(timeStep = 200, trainBegin = 10, trainEnd = 2800):
    #产生dnn网络
    Y = tf.placeholder(tf.float32, shape=[None, 1]) 
    #开盘价变化
    X1 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn1, D1 = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #收盘价变化
    X2 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn2, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #最高价变化
    X3 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn3, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #最低价变化
    X4 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn4, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #换手率变化
    X5 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn5, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #5日均线变化
    X6 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn6, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #10日均线变化
    X7 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn7, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #20日均线变化
    X8 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn8, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #综合输出
    XOut = tf.concat([Dnn1,Dnn2,Dnn3,Dnn4,Dnn5,Dnn6,Dnn7,Dnn8], axis = 1, name = "XOut")
    DnnOut, KeepProb = tamPub1.getMultDnn(XOut, [800, 300, 100, 30, 1], True)
    
    #损失函数
    Loss = tf.square(tf.reduce_sum(DnnOut-Y))*300
    Loss = Loss*Loss
    TrainOp = tf.train.AdamOptimizer(1e-4).minimize(Loss)
    #ResultAi = DnnSoftmax
    ResultAi = DnnOut[0][0] / 5 - 0.1

    #写入模型
    #writer = tf.summary.FileWriter("d://temp//tensorFlow//MnistL1",tf.get_default_graph())  
    #writer.close()

    #准备数据
    global data
    global dataResult
    x1 = data[1:,0] - data[0:-1,0]
    x2 = data[1:,1] - data[0:-1,1]
    x3 = data[1:,2] - data[0:-1,2]
    x4 = data[1:,3] - data[0:-1,3]
    x5 = data[1:,4] - data[0:-1,4]
    x5 = x5 / 5 #换手率
    x6 = tamPub1.getMA(data[:,1], 3) #3日均线
    x6 = x6[1:] - x6[0:-1]
    x7 = tamPub1.getMA(data[:,1], 5) #5日均线
    x7 = x7[1:] - x7[0:-1]
    x8 = tamPub1.getMA(data[:,1], 10) #20日均线
    x8 = x8[1:] - x8[0:-1]
    #常用均线：黄色是5日均线 紫色是10日均线 绿色是20日均线 白色是30日均线 
    #x1 = x1 / 2
    #x2 = x2 / 2
    #x3 = x3 / 2
    #x4 = x4 / 2
    #x5 = x5 + 2
    #x6 = x6 / 1
    #x7 = x7 / 1
    #x8 = x8 / 1
    
    
    #开始训练
    plt.figure()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./stock/stockLSTM2v1.ckpt")
    testMode = True
    for batch in range(1 if testMode else 500):
      diff = 0
      sumDiff = []
      sumAi = []
      sumReal = []
      dataRange = range(2800)
      keepProb = 0.5
      if testMode:
          dataRange = range(2900,3100)
          keepProb = 1
      #for i in range(2800):
      for i in dataRange:
        y = np.reshape(data[i+timeStep-1:i+timeStep, 7], (1,1)) * 5 + 0.5
        if (testMode):
            loss, resultAi = sess.run([Loss, ResultAi], feed_dict = {
                X1:np.reshape(x1[i:i+timeStep], (1,timeStep)),
                X2:np.reshape(x2[i:i+timeStep], (1,timeStep)),
                X3:np.reshape(x3[i:i+timeStep], (1,timeStep)),
                X4:np.reshape(x4[i:i+timeStep], (1,timeStep)),
                X5:np.reshape(x5[i:i+timeStep], (1,timeStep)),
                X6:np.reshape(x6[i:i+timeStep], (1,timeStep)),
                X7:np.reshape(x7[i:i+timeStep], (1,timeStep)),
                X8:np.reshape(x8[i:i+timeStep], (1,timeStep)),
                Y:y,
                KeepProb: keepProb
            })
        else:
            _,loss, resultAi = sess.run([TrainOp, Loss, ResultAi], feed_dict = {
                X1:np.reshape(x1[i:i+timeStep], (1,timeStep)),
                X2:np.reshape(x2[i:i+timeStep], (1,timeStep)),
                X3:np.reshape(x3[i:i+timeStep], (1,timeStep)),
                X4:np.reshape(x4[i:i+timeStep], (1,timeStep)),
                X5:np.reshape(x5[i:i+timeStep], (1,timeStep)),
                X6:np.reshape(x6[i:i+timeStep], (1,timeStep)),
                X7:np.reshape(x7[i:i+timeStep], (1,timeStep)),
                X8:np.reshape(x8[i:i+timeStep], (1,timeStep)),
                Y:y,
                KeepProb: keepProb
            })
        sumDiff = np.hstack((sumDiff, resultAi*100-(y[0][0]/5-0.1)*100))
        sumAi = np.hstack((sumAi, resultAi*100))
        sumReal = np.hstack((sumReal, (y[0][0]/5-0.1)*100))
        
        diff = diff + np.abs(resultAi*100-(y[0][0]/5-0.1)*100)
        draw = False
        if testMode and i == dataRange[-1]:
            draw = True
        elif not testMode and i % 100 == 0:
            draw = True
        if draw:
            plt.subplot(2,1,1)
            plt.cla()
            plt.plot([0,0],[-20,20], color='g')
            plt.plot([0,len(sumDiff)],[0,0], color='g')
            plt.plot([0,len(sumDiff)],[-10,-10], color='g')
            plt.plot([0,len(sumDiff)],[10,10], color='g')
            plt.bar(range(len(sumDiff)), sumDiff, 0.8, color='b')
            plt.title("sumDiff")
            sumDiff = []
            
            plt.subplot(2,1,2)
            plt.cla()
            plt.plot([0,0],[-20,20], color='g')
            plt.plot([0,len(sumAi)],[0,0], color='g')
            plt.plot([0,len(sumAi)],[-10,-10], color='g')
            plt.plot([0,len(sumAi)],[10,10], color='g')
            plt.bar(range(len(sumAi)), sumAi, 0.8, color='b')
            plt.title("sumAi")
            sumAi=[]
            
            plt.bar(range(len(sumReal)), sumReal, 0.2, color='r')
            plt.title("sumAi,sumReal")
            sumReal=[]

            plt.pause(0.001)
            print(i,loss,
                "实际:",(y[0][0]/5-0.1)*100,"%",
                "预测:",resultAi*100,"%"
            )
      print("=======最终整体误差: ", diff/len(dataRange),"%")
      #saver = tf.train.Saver()
      #saver.save(sess, "./stock/stockLSTM2v2.ckpt")
      
def tam_dnn4(testMode = True, batchSize = 40, timeStep = 200, trainBegin = 10, trainEnd = 3900-250-100, testBegin = 3600, testEnd = 3700):
    checkPointFilename = "./stock/stockLStm2Dnn4.ckpt"
    
    #产生dnn网络
    Y = tf.placeholder(tf.float32, shape=[None, 3]) 
    #开盘价变化
    X1 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn1, D1 = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #收盘价变化
    X2 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn2, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #最高价变化
    X3 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn3, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #最低价变化
    X4 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn4, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #换手率变化
    X5 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn5, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #5日均线变化
    X6 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn6, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #10日均线变化
    X7 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn7, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #20日均线变化
    X8 = tf.placeholder(tf.float32, shape=[None, timeStep])
    Dnn8, _ = tamPub1.getMultDnn(X1, [timeStep, 600, 100], False)
    #综合输出
    XOut = tf.concat([Dnn1,Dnn2,Dnn3,Dnn4,Dnn5,Dnn6,Dnn7,Dnn8], axis = 1, name = "XOut")
    DnnOut, KeepProb = tamPub1.getMultDnn(XOut, [800, 300, 100, 30, 3], True)
    
    #损失函数
    Loss = tf.reduce_sum(tf.square(DnnOut-Y))*30
    Loss = Loss*Loss
    TrainOp = tf.train.AdamOptimizer(1e-4).minimize(Loss)
    #ResultAi = DnnSoftmax
    ResultAi = DnnOut / 5 - 0.1

    #写入模型
    #writer = tf.summary.FileWriter("d://temp//tensorFlow//MnistL1",tf.get_default_graph())  
    #writer.close()

    #准备数据
    global data
    global dataResult
    x1 = data[1:,0] - data[0:-1,0]
    x2 = data[1:,1] - data[0:-1,1]
    x3 = data[1:,2] - data[0:-1,2]
    x4 = data[1:,3] - data[0:-1,3]
    x5 = data[1:,4] - data[0:-1,4]
    x5 = x5 / 5 #换手率
    x6 = tamPub1.getMA(data[:,1], 5) #5日均线
    x6 = x6[1:] - x6[0:-1]
    x7 = tamPub1.getMA(data[:,1], 20) #20日均线
    x7 = x7[1:] - x7[0:-1]
    x8 = tamPub1.getMA(data[:,1], 60) #60日均线
    x8 = x8[1:] - x8[0:-1]
    #常用均线：黄色是5日均线 紫色是10日均线 绿色是20日均线 白色是30日均线 
    
    #开始训练
    plt.figure()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    #读取历史训练结果
    saver = tf.train.Saver()
    saver.restore(sess, checkPointFilename)
    print("====> 读取神经网络完成：" + checkPointFilename)
        
    for batch in range(1 if testMode else 500):
        diff = [0,0,0]
        sumAi = np.empty(shape=[0, 3])
        sumReal = np.empty(shape=[0, 3])
        lastDrawPos = -1
        if testMode:
            dataRange = range(testBegin, testEnd)
            keepProb = 1
            batchSize = 1
        else:
            dataRange = range(trainBegin, trainEnd - batchSize)
            keepProb = 0.5
        for i in dataRange:
            y = data[i+timeStep-1:i+timeStep-1+batchSize, 7:10] * 5 + 0.5
            x = np.empty(shape=[8,batchSize,timeStep]) #(8,batchSize,timeStep)
            for j in np.arange(i, i + batchSize):
                x[0][j-i] = x1[j:j+timeStep]
                x[1][j-i] = x2[j:j+timeStep]
                x[2][j-i] = x3[j:j+timeStep]
                x[3][j-i] = x4[j:j+timeStep]
                x[4][j-i] = x5[j:j+timeStep]
                x[5][j-i] = x6[j:j+timeStep]
                x[6][j-i] = x7[j:j+timeStep]
                x[7][j-i] = x8[j:j+timeStep]
            if (testMode):
                loss, resultAi = sess.run([Loss, ResultAi], feed_dict = {
                    X1:x[0],
                    X2:x[1],
                    X3:x[2],
                    X4:x[3],
                    X5:x[4],
                    X6:x[5],
                    X7:x[6],
                    X8:x[7],
                    Y:y,
                    KeepProb: keepProb
                })
            else:
                _,loss, resultAi = sess.run([TrainOp, Loss, ResultAi], feed_dict = {
                    X1:x[0],
                    X2:x[1],
                    X3:x[2],
                    X4:x[3],
                    X5:x[4],
                    X6:x[5],
                    X7:x[6],
                    X8:x[7],
                    Y:y,
                    KeepProb: keepProb
                })
            sumAi = np.vstack((sumAi, np.reshape(resultAi[-1]*100, (3))))
            sumReal = np.vstack((sumReal, (y[-1]/5-0.1)*100))
            diff = diff + np.abs(sumAi[-1] - sumReal[-1])
            draw = False
            if testMode and i == dataRange[-1]:
                draw = True
            elif not testMode and i % 100 == 0:
                draw = True
            if draw:
                lastDrawPos = max(lastDrawPos, dataRange[0])
                plt.subplot(3,1,1)
                plt.cla()
                plt.plot([lastDrawPos, lastDrawPos],[-10,10], color='g')
                plt.plot([lastDrawPos, i],[0,0], color='g')
                plt.plot([lastDrawPos, i],[-10,-10], color='g')
                plt.plot([lastDrawPos, i],[10,10], color='g')
                plt.bar(np.arange(lastDrawPos, i + 1), np.reshape(sumAi[:,0], (len(sumAi))), 0.8, color='b')
                plt.bar(np.arange(lastDrawPos, i + 1), np.reshape(sumReal[:,0], (len(sumReal))), 0.2, color='r')
                plt.title("3Day")
                
                plt.subplot(3,1,2)
                plt.cla()
                plt.plot([lastDrawPos, lastDrawPos],[-10,10], color='g')
                plt.plot([lastDrawPos, i],[0,0], color='g')
                plt.plot([lastDrawPos, i],[-10,-10], color='g')
                plt.plot([lastDrawPos, i],[10,10], color='g')
                plt.bar(np.arange(lastDrawPos, i + 1), np.reshape(sumAi[:,1], (len(sumAi))), 0.8, color='b')
                plt.bar(np.arange(lastDrawPos, i + 1), np.reshape(sumReal[:,1], (len(sumReal))), 0.2, color='r')
                plt.title("5Day")

                plt.subplot(3,1,3)
                plt.cla()
                plt.plot([lastDrawPos, lastDrawPos],[-10,10], color='g')
                plt.plot([lastDrawPos, i],[0,0], color='g')
                plt.plot([lastDrawPos, i],[-10,-10], color='g')
                plt.plot([lastDrawPos, i],[10,10], color='g')
                plt.bar(np.arange(lastDrawPos, i + 1), np.reshape(sumAi[:,2], (len(sumAi))), 0.8, color='b')
                plt.bar(np.arange(lastDrawPos, i + 1), np.reshape(sumReal[:,2], (len(sumReal))), 0.2, color='r')
                plt.title("10Day")
            
                print(i,loss, ",3日:", diff[0]/(i-dataRange[0]+1),"%,5日:", diff[1]/(i-dataRange[0]+1), "%,10日:", diff[2]/(i-dataRange[0]+1),"%")

                sumAi = np.empty(shape=[0, 3])
                sumReal = np.empty(shape=[0, 3])
                lastDrawPos = i + 1
                plt.pause(0.001)
        print("=======第", batch,"轮整体误差,3日:", diff[0]/len(dataRange),"%,5日:", diff[1]/len(dataRange), "%,10日:", diff[2]/len(dataRange),"%")
        if not testMode: 
            saver = tf.train.Saver()
            #saver.save(sess, checkPointFilename)
            #print("====> 保存神经网络完成：" + checkPointFilename)

tam_dnn4()
#tam_lstm()
plt.show()


