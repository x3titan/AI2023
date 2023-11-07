#coding=utf-8
#采用LSTM神经网络预测股票数据
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import csv

#定义常量
rnn_unit=10       #hidden layer units
input_size=5
output_size=1
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
#时间    1开盘    2最高    3最低    4收盘    5涨幅    6振幅    7总手    8金额    9换手%    10成交次数    11明日收盘
#2000-12-12,二    4.96    7.36    4.88    6.09    --    --    312,239,700    0    69.39    0    5.71
#2000-12-13,三    5.81    5.94    5.69    5.71    -6.24%    4.11%    95,947,400    0    21.32    0    5.58
filename = "./stock/600019.csv"
print("=======>> 开始载入股票数据：%s"%(filename))
csv_reader = csv.reader(open(filename, encoding="gb2312"))
#开盘，收盘，最高，最低，换手率，明日收盘
data =  np.empty(shape=[0, 6])
data0 =  np.empty(shape=[0, 6])
i = 0
for row in csv_reader:
    if i > 1: #跳过题头和开盘第一天
        data0 = np.vstack((data0, [float(row[1]), float(row[4]), float(row[3]), float(row[2]), float(row[9]), float(row[11])]))
        data = np.vstack((data, [float(row[1])/100, float(row[4])/100, float(row[3])/100, float(row[2])/100, float(row[9])/100, float(row[12])*5+0.5]))
    i = i + 1
print("=======>> 股票数据载入完毕，共载入%d条记录"%(len(data)))
data = data0

#获取训练集
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=2800):
    batch_index=[]
    data_train=data[train_begin:train_end]
    data_mean=np.mean(data_train,axis=0)
    data_std=np.std(data_train,axis=0)
    #normalized_train_data=data_train
    normalized_train_data=(data_train-data_mean)/data_std  #标准化
    print("normalized_train_data", np.shape(normalized_train_data))
    train_x,train_y=[],[]   #训练集 
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:5]
       y=normalized_train_data[i:i+time_step,5,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y,data_train



#获取测试集
def get_test_data(time_step=20,test_begin=2800):
    data_test=data[test_begin:test_begin+200]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample 
    test_x,test_y=[],[]  
    i = 0
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:5]
       y=normalized_test_data[i*time_step:(i+1)*time_step,5]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:5]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,5]).tolist())
    return mean,std,test_x,test_y



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
    return pred, output_rnn, final_states

def lstm2(X, model="lstm", num_layers=1):     
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]

    if model == "rnn":  
        cell_fun = tf.nn.rnn_cell.BasicRNNCell  
    elif model == "gru":  
        cell_fun = tf.nn.rnn_cell.GRUCell  
    elif model == "lstm":  
        cell_fun = tf.nn.rnn_cell.BasicLSTMCell  

    cell = cell_fun(rnn_unit, state_is_tuple=True)  
    cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)  

    initial_state = cell.zero_state(batch_size, tf.float32)  

    output_rnn, final_states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state)  
    #output = tf.reshape(outputs,[-1, rnn_size])  

    #logits = tf.matmul(output, softmax_w) + softmax_b  
    #probs = tf.nn.softmax(logits)  
    #return logits, last_state, probs, cell, initial_state 
    print("output_rnn", np.shape(output_rnn))
    print("final_states",final_states)
    print("final_states", np.shape(final_states[0][1]))
    return final_states[0][0]

#——————————————————训练模型——————————————————
def train_lstm(batch_size=80,time_step=20,train_begin=10,train_end=2800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size]) 
    batch_index,train_x,train_y,_=get_train_data(batch_size,time_step,train_begin,train_end)
    mean,std,test_x,test_y=get_test_data(time_step)
    print("trainx:", np.shape(train_x))
    print("trainy:", np.shape(train_y))
    print("test_x:", np.shape(test_x))
    print("test_y:", np.shape(test_y))
    pred,_,_=lstm(X)
    print("pred: ", np.shape(pred))
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    #准确率
    acc = tf.reshape(pred,[-1]) / tf.reshape(Y, [-1]) - 1
    
    #saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint()    
    plt.figure()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        #重复训练10000次
        loss_ = None
        for i in range(50):
            for step in range(len(batch_index)-1):
                _,loss_,acc_=sess.run([train_op,loss, acc],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            #if i % 10==0:
            #    print("保存模型：",saver.save(sess,'stock2.model',global_step=i))
            plt.cla()
            plt.plot([0, len(acc_)], [0,0], color='g',  linewidth = 0.6)
            plt.plot([0, 0], [-10, 10], color='g',  linewidth = 0.6)
            plt.plot(acc_,  color='b',  linewidth = 0.6)
            plt.pause(0.001)

        #参数恢复
        #module_file = tf.train.latest_checkpoint()
        #saver.restore(sess, module_file) 
        test_predict=[]
        for step in range(len(test_x)-1):
            prob=sess.run(pred,feed_dict={X:[test_x[step]]})   
            predict=prob.reshape((-1))
            test_predict.extend(predict)
        test_y=np.array(test_y)*std[5]+mean[5]
        test_predict=np.array(test_predict)*std[5]+mean[5]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b', linewidth = 0.2)
        plt.plot(list(range(len(test_y))), test_y,  color='r', linewidth = 0.2)
        plt.show()


#train_lstm()

#————————————————预测模型————————————————————
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    print(np.shape(test_y))
    #pred,_=lstm(X)     
    global pred
    #saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        #module_file = tf.train.latest_checkpoint()
        #saver.restore(sess, module_file) 
        test_predict=[]
        for step in range(len(test_x)-1):
          #prob = pred.eval(feed_dict={X:[test_x[step]]})
          prob=sess.run(pred,feed_dict={X:[test_x[step]], Y: [test_y[step]]})   
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b', linewidth = 0.2)
        plt.plot(list(range(len(test_y))), test_y,  color='r', linewidth = 0.2)
        plt.show()

#prediction() 


print("————————————————计算并预测————————————————————")
def tam_lstm(batch_size=80,time_step=200,train_begin=10,train_end=2800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size]) 
    batch_index,train_x,train_y,data_src=get_train_data(batch_size,time_step,train_begin,train_end)
    print("trainx:", np.shape(train_x))
    print("trainy:", np.shape(train_y))
    #pred,output_rnn,final_states=lstm(X)
    pred,output_rnn,final_states=lstm2(X)
    #损失函数
    #loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    diff = tf.reshape(pred,[-1])[time_step-1]-tf.reshape(Y, [-1])[time_step-1]
    loss=tf.square(diff)
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    #准确率
    p1 = tf.reshape(pred,[-1])[time_step-1]/5-0.1
    p2 = tf.reshape(Y, [-1])[time_step-1]/5-0.1
    acc = p1-p2
    
    plt.figure()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(train_end-train_begin-time_step):
        _,_loss,_acc,_pred,a1,a2=sess.run([train_op,loss, acc, pred, output_rnn, final_states],feed_dict={
            X:train_x[i:i+1],
            Y:train_y[i:i+1]
        })
        print("output_rnn", np.shape(a1))
        print("final_states", np.shape(a2))

        #print(np.shape(_finalState))
        #0开盘，1收盘，2最高，3最低，4换手率，5明日收盘
        #print("pred_",np.shape(_pred))
        #print("train_y",np.shape(train_y))
        #print("train_y[i:i+1]", np.shape(train_y[i:i+1]))
        print(
            "序号:",i+train_begin+time_step+2,
            ", 涨幅:",(train_y[i][time_step-1][0]/5-0.1)*100,"%",
            ",预测:",(_pred[time_step-1][0]/5-0.1)*100,"%",
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
        v1 = (_pred[time_step-1][0]/5-0.1)*100
        v2 = (train_y[i][time_step-1][0]/5-0.1)*100
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

print("————————————————计算并预测————————————————————")
def tam_lstm2(batch_size=80,time_step=200,train_begin=10,train_end=2800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size]) 
    batch_index,train_x,train_y,data_src=get_train_data(batch_size,time_step,train_begin,train_end)
    print("trainx:", np.shape(train_x))
    print("trainy:", np.shape(train_y))
    #pred,output_rnn,final_states=lstm(X)
    final_states=lstm2(X)
    print("",np.shape(final_states))
    #损失函数
    diff = final_states-tf.reshape(Y, [-1])[time_step-1]
    loss=tf.square(diff)
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    #准确率
    p1 = final_states/5-0.1
    p2 = tf.reshape(Y, [-1])[time_step-1]/5-0.1
    acc = p1-p2

    
    plt.figure()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(train_end-train_begin-time_step):
        _,_loss,_acc,_final_states=sess.run([train_op,loss,acc,final_states],feed_dict={
            X:train_x[i:i+1],
            Y:train_y[i:i+1]
        })
        print(final_states)
        
        #print(np.shape(_finalState))
        #0开盘，1收盘，2最高，3最低，4换手率，5明日收盘
        #print("pred_",np.shape(_pred))
        #print("train_y",np.shape(train_y))
        #print("train_y[i:i+1]", np.shape(train_y[i:i+1]))
        print(
            "序号:",i+train_begin+time_step+2,
            ", 涨幅:",(train_y[i][time_step-1][0]/5-0.1)*100,"%",
            #",预测:",(_pred[time_step-1][0]/5-0.1)*100,"%",
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
        v1 = (final_states/5-0.1)*100
        v2 = (train_y[i][time_step-1][0]/5-0.1)*100
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
        
train_lstm()
#tam_lstm2()
plt.show()













