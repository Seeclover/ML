import math
import numpy as np
import random
import pandas as pd

correct_c = np.zeros((3, 1))

# 訓練模型並找到分類錯誤最少的權重參數 c
for l in range(10000):
    seed, n0, m = 1, 100, 2
    ans = 100
    A = np.random.rand(m, 3)  # 使用隨機值初始化 A
    b = np.random.rand(1, 3)  # 使用隨機值初始化 b
    c = np.random.rand(3, 1)  # 使用隨機值初始化 c

    random.seed(seed)
    dataPointSe = np.random.rand(n0, m)  # 使用隨機值初始化 dataPointSet
    dataPointSet = dataPointSe * 10 - 5
    label = np.zeros((n0, 1))

    # 生成標籤
    for i in range(n0):
        if dataPointSet[i, 0]**2 + dataPointSet[i, 1]**2 >= 5:
            label[i, 0] = 1
        else:
            label[i, 0] = -1

    learning_rate = 0.01

    # 訓練迴圈
    for epoch in range(1000):
        z = dataPointSet.dot(A) + b

        # 計算梯度
        sech_squared = 1 / np.cosh(z)**2
        gradient = dataPointSet.T.dot(sech_squared.dot(c))
        gradient_b = np.sum(sech_squared.dot(c), axis=0)

        # 更新權重參數 A 和 b
        A -= learning_rate * gradient
        b -= learning_rate * gradient_b

    # 預測並計算錯誤率
    z = dataPointSet.dot(A) + b
    loss = -z.dot(c)
    guess = np.sign(loss)  # 二元分類，假設 1 和 -1 是兩個類別
    err = np.sum(guess != label)  # 計算錯誤分類的點的數量

    # 找到錯誤率最低的 c
    if err < ans:
        ans = err
        correct_c = c

c = correct_c

random.seed(seed)
dataPointSe = np.random.rand(n0, m)  # 使用隨機值初始化 dataPointSet
dataPointSet = dataPointSe * 10 - 5
label = np.zeros((n0, 1))

# 生成標籤
for i in range(n0):
    if dataPointSet[i, 0]**2 + dataPointSet[i, 1]**2 >= 5:
        label[i, 0] = 1
    else:
        label[i, 0] = -1

learning_rate = 0.01

# 再次訓練模型
for epoch in range(10000):
    z = dataPointSet.dot(A) + b

    # 計算梯度
    sech_squared = 1 / np.cosh(z)**2
    gradient = dataPointSet.T.dot(sech_squared.dot(c))
    gradient_b = np.sum(sech_squared.dot(c), axis=0)

    # 更新權重參數 A 和 b
    A -= learning_rate * gradient
    b -= learning_rate * gradient_b

z = dataPointSet.dot(A) + b
loss = -z.dot(c)
guess = np.sign(loss)  # 二元分類，假設 1 和 -1 是兩個類別
err = np.sum(guess != label)  # 計算錯誤分類的點的數量
print(err)
