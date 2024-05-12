#预测乘法，三位数乘以三位数，结果为6位数
#完全独立的6张网络，分别预测6位数字

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pub.tamPub1 as tamPub1
import sys
import os
import shutil

def generateGoldenDnn(inputSize, outputSize, inputArg = "relu", outputArg = "softmax"):
    aiArray = [inputSize]
    decrementFactor = 0.62  # 黄金分割
    while round(aiArray[-1] * decrementFactor) > outputSize:
        aiArray.append(round(aiArray[-1] * decrementFactor))
    aiArray.append(outputSize)
    dnn = tamPub1.getMultiDnnV3(aiArray, inputArg, outputArg)
    return dnn

def generateLayerDnn(inputSize, outputSize, layerCount):
    aiArray = [inputSize]
    dy = (outputSize - inputSize) / (layerCount + 1)
    for i in range(1, layerCount + 1):
        aiArray.append(round(inputSize + dy * i))
    aiArray.append(outputSize)
    dnn = tamPub1.getMultiDnnV3(aiArray, "relu", "sigmoid")
    # dnn.compile(optimizer='adam',
    #    loss='sparse_categorical_crossentropy', # 适用于分类问题
    #    metrics=['accuracy'])
    return dnn


# 获取整数(intNumber)的某位(n)的数字，0表示最低位的数字
def getIntDigit(intNumber: int, n: int):
    return (intNumber // 10**n) % 10


# physical_devices = tf.config.list_physical_devices('GPU')
physical_devices = tf.config.experimental.list_physical_devices()

print("Num GPUs:", len(physical_devices))
print("Available devices:", physical_devices)

# 找到 DML 设备
dml_devices = [d for d in physical_devices if "DML" in d.name]

# 设置 DML 设备为可见（假设我们想使用第一个 DML 设备）
if dml_devices:  # 确保有 DML 设备可用
    print("Use DML device 0 ", dml_devices[0])
    tf.config.experimental.set_visible_devices(dml_devices[0], "DML")
else:
    print("No DML devices found")

# if physical_devices:
#     # 设置 TensorFlow 可见的设备为 GPU（或 CPU）
#     tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
#     print("使用GPU")
# else:
#     # 如果没有 GPU，则设置 TensorFlow 可见的设备为 CPU
#     tf.config.experimental.set_visible_devices([], 'CPU')
#     print("警告：找不到GPU使用CPU")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
dnns = []
for i in range(6):
    #dnn = generateGoldenDnn(60, 10, "relu", "softmax")
    dnn = Sequential([
        Dense(37, activation='relu', input_shape=(60,)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(23, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(14, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(10, activation='softmax')
    ])
    dnns.append(dnn)
    dnn.compile(optimizer='adam',
        loss='categorical_crossentropy',  # 更换为适合多标签分类的损失函数
        metrics=['accuracy'])

# dnn.compile(optimizer=Adam(learning_rate=0.0005),
#             loss='categorical_crossentropy',
#             metrics=['accuracy'],
#             loss_weights=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2])  # 根据位的重要性分配权重
dnns[0].summary()


currentDir = os.path.dirname(os.path.abspath(__file__))
file_name_with_extension = os.path.basename(__file__)
file_name_without_extension = os.path.splitext(file_name_with_extension)[0]

saveFileNames = [];
for i in range(len(dnns)):
    saveFileNames.append(currentDir + "\\dnnSave\\" + file_name_without_extension + f"_DIG{i}.tmp")
if os.path.exists(saveFileNames[0]):
    for dnnIndex in range(len(dnns)):
        dnns[dnnIndex] = tf.keras.models.load_model(saveFileNames[dnnIndex])
        print(f"载入模型文件: {saveFileNames[dnnIndex]}")
else:
    print(f"文件 {saveFileNames[0]} 不存在，无需删除。")


# 训练模型
loss_history = []
batchSize = 1000
#初始化图形
#plt.ion()
fig, axs = plt.subplots(3, 2, figsize=(20, 10))  # 创建六个子图
axs = axs.flatten()  # 将axs数组扁平化，便于索引
lines = [ax.plot([], [])[0] for ax in axs]  # 创建每个子图的line对象
dig = 0
for ax in axs:
    ax.set_title(f'DIG {dig}')
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    dig+=1
plt.tight_layout()
plt.show(block=False)

for i in range(0, 5000):
    tIn = np.zeros((batchSize, 60), dtype=float)
    tOut =[np.zeros((batchSize, 10), dtype=float) for _ in range(6)]
    for batchIndex in range(0, batchSize):  # 堆叠一批batchSize个乘法的例子
        a = np.random.randint(0, 999)
        b = np.random.randint(0, 999)
        for n in range(0, 3):
            tIn[batchIndex, getIntDigit(a, n) + n * 10] = 1
        for n in range(0, 3):
            tIn[batchIndex, getIntDigit(b, n) + n * 10 + 30] = 1
        for n in range(0, 6):
            tOut[n][batchIndex, getIntDigit(a * b, n)] = 1

    print(f"产生数据完成，开始训练，第{i}批")

    # 0：不显示训练进度信息。
    # 1：显示一个进度条，每个训练周期结束时显示一行训练进度信息。
    # 2：每个训练周期结束时显示一行训练进度信息，包括每个 epoch 的平均损失和度量指
    #dnn.fit(tIn, tOut, epochs=10, verbose=0)
    hisItem = []
    for dnnIndex in range(len(dnns)):
        dnns[dnnIndex].fit(tIn, tOut[dnnIndex],
            epochs=100,
            batch_size=batchSize,
            verbose=0)
        loss = dnns[dnnIndex].evaluate(tIn, tOut[dnnIndex])
        hisItem.append(loss[0])
    loss_history.append(hisItem)
    print(f"Loss: {hisItem}")
    
    #图形显示loss
    for i in range(6):
        lines[i].set_xdata(range(len(loss_history)))
        lines[i].set_ydata([row[i] for row in loss_history])
        axs[i].relim()
        axs[i].autoscale_view(True, True, True)

    fig.canvas.draw()
    fig.canvas.flush_events()
    
    # 根据实际模型文件名修改
    for dnnIndex in range(len(dnns)):
        dnns[dnnIndex].save(saveFileNames[dnnIndex])

    # 验证模型
    tIn = np.zeros((5, 60), dtype=float)
    tOutReal = np.zeros((5, 1), dtype=float)
    tSrc = np.zeros((5, 2), dtype=float)
    for stackIndex in range(0, 5):  # 仅验证5个
        a = np.random.randint(0, 999)
        b = np.random.randint(0, 999)
        tSrc[stackIndex, 0] = a
        tSrc[stackIndex, 1] = b
        for n in range(0, 3):
            tIn[stackIndex, getIntDigit(a, n) + n * 10] = 1
        for n in range(0, 3):
            tIn[stackIndex, getIntDigit(b, n) + n * 10 + 30] = 1
    pred = []
    for dnnIndex in range(len(dnns)):
        pred.append(dnns[dnnIndex].predict(tIn))

    for stackIndex in range(0, 5):
        for n in range(0, 6):
            tOutReal[stackIndex, 0] += (10**n) * np.argmax(
                pred[n][stackIndex, :]
            )
    for stackIndex in range(0, 5):  # 仅验证5个
        print(
            f"a={tSrc[stackIndex,0]},b={tSrc[stackIndex,1]},正确结果{int(tSrc[stackIndex,0]*tSrc[stackIndex,1])},"
            + f"预测结果{int(tOutReal[stackIndex,0])},差值{int(tOutReal[stackIndex,0]-tSrc[stackIndex,0]*tSrc[stackIndex,1])},"
            + f"差值百分比{(tOutReal[stackIndex,0]/(tSrc[stackIndex,0]*tSrc[stackIndex,1])-1)*100:.2f}%"
        )


# 使用模型进行预测
# predictions = model.predict(X_train)
# print("pred:")
# print(pred)
print("AI2023")