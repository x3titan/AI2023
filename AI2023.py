import tensorflow as tf
import numpy as np
import pub.tamPub1 as tamPub1
import sys
import os
import shutil

def generateDnn(timeStep):
    #产生DNN网络
    Y = tf.placeholder(tf.float32, shape=[None, 2])
    X = tf.placeholder(tf.float32, shape=[None, timeStep * 10 + 48 * 10 * 3])
    #DnnOut, KeepProb = tamPub1.getMultDnn(X, [2400, 1380, 792, 475, 285, 171,
    #102, 61, 36, 22, 12, 7, 4], True) #采用黄金分割方案
    DnnOut, KeepProb = tamPub1.getMultDnn(X, [2440, 4000, 4000, 2800, 1940, 1164, 690, 475, 285, 171, 102, 61, 36, 22, 12, 7, 4, 2], True)      #采用黄金分割方案
    
    #损失函数
    Loss = tf.reduce_sum(tf.square(DnnOut - Y)) * 30
    #Loss = Loss * Loss
    TrainOp = tf.train.AdamOptimizer(0.3).minimize(Loss) #1e-5
    ResultAi = DnnOut
    return X,Y,KeepProb, TrainOp,Loss,ResultAi

def generateGoldenDnn(inputSize, outputSize):
    aiArray = [inputSize]
    decrementFactor = 0.62  #黄金分割
    while round(aiArray[-1] * decrementFactor) > outputSize:
        aiArray.append(round(aiArray[-1] * decrementFactor))
    aiArray.append(outputSize)
    
    dnn = tamPub1.getMultiDnnV2(aiArray)
    #dnn.compile(optimizer='adam',
    #    loss='sparse_categorical_crossentropy', # 适用于分类问题
    #    metrics=['accuracy'])
    return dnn

#获取整数(intNumber)的某位(n)的数字，0表示最低位的数字
def getIntDigit(intNumber: int, n: int):
    return (intNumber // 10 ** n) % 10


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    # 设置 TensorFlow 可见的设备为 GPU（或 CPU）
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
    print("使用GPU")
else:
    # 如果没有 GPU，则设置 TensorFlow 可见的设备为 CPU
    tf.config.experimental.set_visible_devices([], 'CPU')
    print("警告：找不到GPU使用CPU")

#====================================================================
#999*999的测试乘法模型
#输入10*6=60
#输出1, 重整化1000*1000=1000000为1
dnn = generateGoldenDnn(60,1)
dnn.compile(optimizer='adam', loss='mean_squared_error')

currentDir = os.path.dirname(os.path.abspath(__file__))

file_path = currentDir + r"\dnnSave\ai2023.tmp"
if os.path.exists(file_path):
    #shutil.rmtree(file_path);
    print(f"文件 {file_path} 已被删除。")
else:
    print(f"文件 {file_path} 不存在，无需删除。")

# 训练模型
losses = []
for i in range(0,5000):
    tIn = np.zeros((1000, 60), dtype=float)
    tOut = np.zeros((1000, 1), dtype=float)
    for stackCount in range(0,1000): #堆叠一批100个乘法的例子
        a = np.random.randint(0, 999)
        b = np.random.randint(0, 999)
        for n in range(0,3):
            tIn[stackCount, getIntDigit(a, n) + n * 10] = 1
        for n in range(0,3):
            tIn[stackCount, getIntDigit(b, n) + n * 10 + 30] = 1
        tOut[stackCount, 0] = a * b / 1000 / 1000
    
    print(f'产生数据完成，开始训练，第{i}批')
    if os.path.exists(file_path):
        dnn = tf.keras.models.load_model(file_path)
    
    #0：不显示训练进度信息。
    #1：显示一个进度条，每个训练周期结束时显示一行训练进度信息。
    #2：每个训练周期结束时显示一行训练进度信息，包括每个 epoch 的平均损失和度量指
    dnn.fit(tIn, tOut, epochs=10, verbose=0)
    # 评估模型（使用训练数据）
    #loss, accuracy = dnn.evaluate(tIn, tOut)
    #print(f"Loss: {loss}, Accuracy: {accuracy}")
    loss = dnn.evaluate(tIn, tOut)
    losses.append(loss)
    print(f"Loss: {loss}")
    dnn.save(file_path)  # 根据实际模型文件名修改

    #验证模型
    tIn = np.zeros((5, 60), dtype=float)
    tOut = np.zeros((5, 1), dtype=float)
    tSrc = np.zeros((5, 2), dtype=float)
    for stackCount in range(0,5): #仅验证5个
        a = np.random.randint(0, 999)
        b = np.random.randint(0, 999)
        tSrc[stackCount, 0] = a
        tSrc[stackCount, 1] = b
        for n in range(0,3):
            tIn[stackCount, getIntDigit(a, n) + n * 10] = 1
        for n in range(0,3):
            tIn[stackCount, getIntDigit(b, n) + n * 10 + 30] = 1
        tOut[stackCount, 0] = a * b / 1000 / 1000
    predictions = dnn.predict(tIn)
    for stackCount in range(0,5): #仅验证5个
        print(f'a={tSrc[stackCount,0]},b={tSrc[stackCount,1]},正确结果{int(tSrc[stackCount,0]*tSrc[stackCount,1])},'+
              f'预测结果{int(predictions[stackCount,0]*1000000)},差值{int(predictions[stackCount,0]*1000000-tSrc[stackCount,0]*tSrc[stackCount,1])},'+
              f'差值百分比{(predictions[stackCount,0]*1000000/(tSrc[stackCount,0]*tSrc[stackCount,1])-1)*100:.2f}%')


# 使用模型进行预测
#predictions = model.predict(X_train)
#print("Predictions:")
#print(predictions)
print("AI2023");