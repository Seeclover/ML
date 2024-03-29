import math
import matplotlib.pyplot as plt
import numpy as np
import random 
import pandas as pd

# 設置隨機種子
seed, n0, m = 1, 30, 2
w = [-1 + 2 * random.random(), random.random()]
b = [random.random(), random.random()]

random.seed(seed)
dataPointSet = []
i = 0
label = []

# 生成隨機數據點
while i < n0:
    dataPointSet.append([])
    for j in range(m):
        dataPointSet[-1].append(-10 + random.random() * 20)
    x = dataPointSet[-1]
    if w[0] * x[0] + w[1] * x[1] >= b[1]:
        label.append(1)
        i = i + 1
    elif w[0] * x[0] + w[1] * x[1] <= b[0]:
        label.append(-1)
        i = i + 1
    else:
        dataPointSet.remove(x)

# 定義計算步長的函數
def determineStepSize(x, d, r):
    n = len(d)
    stepSize = float("inf")
    for i in range(n):
        if d[i] < 0 and stepSize > -x[i] / d[i]:
            stepSize = -x[i] / d[i]
    stepSize = min([1, r * stepSize])
    return stepSize

# 初始化參數
iterationNo = []
iterationX = []
iterationX5 = []
iterationX6 = []
iterationX7 = []
iterationX8 = []

m = n0
n = 2 * n0 + 6

# 初始化變量
A0 = np.zeros((m, 6))
x0 = np.zeros((n, 1))
y0 = np.zeros((m, 1))
s0 = np.zeros((n, 1))

# 隨機初始化變量值
for i in range(n):
    x0[i] = random.randint(1, 10)
for i in range(m):
    y0[i] = random.randint(1, 10)
for i in range(n):
    s0[i] = random.randint(1, 10)

# 構建線性不等式約束矩陣A
for i in range(m):
    A0[i, 0] = dataPointSet[i][0] * label[i]
    A0[i, 1] = -dataPointSet[i][0] * label[i]
    A0[i, 2] = dataPointSet[i][1] * label[i]
    A0[i, 3] = -dataPointSet[i][1] * label[i]
    A0[i, 4] = label[i]
    A0[i, 5] = -label[i]

A1 = np.identity(m)
A2 = np.identity(m) * -1
A = np.hstack((A0, A2, A1))

b = np.ones((m, 1))
c = np.zeros((n, 1))

# 設置c向量
c[0] = x0[0] - x0[1]
c[1] = x0[1] - x0[0]
c[2] = x0[2] - x0[3]
c[3] = x0[3] - x0[2]
for i in range(m):
    c[i + n - m] = 1000000

# 初始化mu值
mu_0 = 10
r = 0.99
epsilon = 1e-6
mu = []
mu.append(mu_0)

e = np.ones((n, 1))

k = 0
# 主循環
while mu[k] > epsilon * mu_0:
    solution = np.vstack((x0, y0, s0))
    X = np.diag(x0[:, 0])
    S = np.diag(s0[:, 0])
    r1 = b - A.dot(x0)
    r2 = c - A.transpose().dot(y0) - s0
    r3 = mu[k] * e - X.dot(S).dot(e)
    residual = np.vstack((r1, r2, r3))

    A1 = np.hstack((A, np.zeros((m, m)), np.zeros((m, n))))
    A2 = np.hstack((np.zeros((n, n)), A.transpose(), np.identity(n)))
    A3 = np.hstack((S, np.zeros((n, m)), X))
    barA = np.vstack((A1, A2, A3))

    NewtonDirection = np.linalg.inv(barA).dot(residual)
    delta_x = NewtonDirection[0:n, 0].reshape(n, 1)
    delta_y = NewtonDirection[n:n + m, 0].reshape(m, 1)
    delta_s = NewtonDirection[n + m:2 * n + m, 0].reshape(n, 1)

    stepSizeX = determineStepSize(x0, delta_x, r)
    stepSizeS = determineStepSize(s0, delta_s, r)
    x0 = x0 + stepSizeX * delta_x
    y0 = y0 + stepSizeS * delta_y
    s0 = s0 + stepSizeS * delta_s

    c[0] = x0[0] - x0[1]
    c[1] = x0[1] - x0[0]
    c[2] = x0[2] - x0[3]
    c[3] = x0[3] - x0[2]

    mu.append((1 / 10) * x0.transpose().dot(s0) / n)

    iterationNo.append(k)
    for i in range(m):
        iterationX.append(x0[i, 0])
    iterationX5.append(r1.transpose().dot(r1)[0, 0])
    iterationX6.append(r2.transpose().dot(r2)[0, 0])
    iterationX7.append(x0.transpose().dot(s0)[0, 0])
    iterationX8.append(c.transpose().dot(x0)[0, 0])
    k = k + 1

# 輸出結果
Table = {"k": iterationNo}
if n <= 4:
    for i in range(n):
        Table["$x_{" + str(i) + "}$"] = iterationX[i]
Table["norm($r_1$)"] = iterationX5
Table["norm($r_2$)"] = iterationX6
Table["$x^{t}s$"] = iterationX7
Table["$c^tx$"] = iterationX8

Table = pd.DataFrame(Table).set_index("k")

# 繪製散))
x0 = np.zeros((n, 1))
y0 = np.zeros((m, 1))
s0 = np.zeros((n, 1))

# 隨機初始化變量值
for i in range(n):
    x0[i] = random.randint(1, 10)

for i in range(m):
    y0[i] = random.randint(1, 10)

for i in range(n):
    s0[i] = random.randint(1, 10)

# 構建線性規劃問題
for i in range(m):
    A0[i, 0] = dataPointSet[i][0] * label[i]
    A0[i, 1] = -dataPointSet[i][0] * label[i]
    A0[i, 2] = dataPointSet[i][1] * label[i]
    A0[i, 3] = -dataPointSet[i][1] * label[i]
    A0[i, 4] = label[i]
    A0[i, 5] = -label[i]

A1 = np.identity(m)
A2 = np.identity(m) * -1
A = np.hstack((A0, A2, A1))

b = np.ones((m, 1))
c = np.zeros((n, 1))

# 初始化 c
c[0] = x0[0] - x0[1]
c[1] = x0[1] - x0[0]
c[2] = x0[2] - x0[3]
c[3] = x0[3] - x0[2]

for i in range(m):
    c[i + n - m] = 1000000

mu_0 = 10
r = 0.99
epsilon = 1e-6

# 迭代求解
mu = []
mu.append(mu_0)

e = np.ones((n, 1))

k = 0
while mu[k] > epsilon * mu_0:

    solution = np.vstack((x0, y0, s0))
    X = np.diag(x0[:, 0])
    S = np.diag(s0[:, 0])

    r1 = b - A.dot(x0)
    r2 = c - A.transpose().dot(y0) - s0
    r3 = mu[k] * e - X.dot(S).dot(e)
    residual = np.vstack((r1, r2, r3))

    A1 = np.hstack((A, np.zeros((m, m)), np.zeros((m, n))))
    A2 = np.hstack((np.zeros((n, n)), A.transpose(), np.identity(n)))
    A3 = np.hstack((S, np.zeros((n, m)), X))
    barA = np.vstack((A1, A2, A3))

    NewtonDirection = np.linalg.inv(barA).dot(residual)
    delta_x = NewtonDirection[0:n, 0].reshape(n, 1)
    delta_y = NewtonDirection[n:n + m, 0].reshape(m, 1)
    delta_s = NewtonDirection[n + m:2 * n + m, 0].reshape(n, 1)

    stepSizeX = determineStepSize(x0, delta_x, r)
    stepSizeS = determineStepSize(s0, delta_s, r)
    x0 = x0 + stepSizeX * delta_x
    y0 = y0 + stepSizeS * delta_y
    s0 = s0 + stepSizeS * delta_s

    c[0] = x0[0] - x0[1]
    c[1] = x0[1] - x0[0]
    c[2] = x0[2] - x0[3]
    c[3] = x0[3] - x0[2]

    mu.append((1 / 10) * x0.transpose().dot(s0) / n)

    iterationNo.append(k)
    for i in range(m):
        iterationX.append(x0[i, 0])
    iterationX5.append(r1.transpose().dot(r1)[0, 0])
    iterationX6.append(r2.transpose().dot(r2)[0, 0])
    iterationX7.append(x0.transpose().dot(s0)[0, 0])
    iterationX8.append(c.transpose().dot(x0)[0, 0])
    k = k + 1

# 構建表格
Table = {"k": iterationNo}
if n <= 4:
    for i in range(n):
        Table["$x_{" + str(i) + "}$"] = iterationX[i]
Table["norm($r_1$)"] = iterationX5
Table["norm($r_2$)"] = iterationX6
Table["$x^{t}s$"] = iterationX7
Table["$c^tx$"] = iterationX8
Table = pd.DataFrame(Table).set_index("k")

# 繪製散點圖
n = len(dataPointSet)
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([dataPointSet[i][0] for i in range(n) if label[i] == -1], [dataPointSet[i][1] for i in range(n) if label[i] == -1], 'o', label='$-1$')
ax.plot([dataPointSet[i][0] for i in range(n) if label[i] == 1], [dataPointSet[i][1] for i in range(n) if label[i] == 1], 'x', label='$1$')
ax.axhline(y=0, xmin=-4, xmax=16)
ax.axvline(x=0, ymin=0, ymax=16)
plt.xlim([-15, 15])
plt.ylim([-15, 15])
ax.set_xlabel('$x_0$', fontsize=18)
ax.set_ylabel('$x_1$', fontsize=18)
ax.set_title('SVM', fontsize=18)
ax.legend(loc='upper right', fontsize=18)
ax.grid()

# 繪製分類線
w1 = x0[0] - x0[1]
w2 = x0[2] - x0[3]
b = x0[4] - x0[5]
x = np.linspace(15, -15, 100)
y = -w1 / w2 * x - b / w2
y1 = -w1 / w2 * x - b / w2 + 1 / w2
y2 = -w1 / w2 * x - b / w2 - 1 / w2
ax.plot(x, y)
ax.plot(x, y1)
ax.plot(x, y2)

plt.show()
plt.savefig('output.png')
