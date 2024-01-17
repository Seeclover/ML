import math
import numpy as np
import random
import pandas as pd

correct_c=np.zeros((3,1))

for l in range(10000):
    seed, n0, m = 1, 100, 2
    ans=100
    A = np.random.rand(m, 3)  # Initialize A with random values
    b = np.random.rand(1, 3)  # Initialize b with random values
    c = np.random.rand(3, 1)  # Initialize c with random values

    random.seed(seed)
    dataPointSe = np.random.rand(n0, m)  # Initialize dataPointSet with random values
    dataPointSet = dataPointSe * 10 - 5
    label = np.zeros((n0, 1))

    for i in range(n0):
        if dataPointSet[i, 0]**2 + dataPointSet[i, 1]**2 >= 5:
            label[i, 0] = 1
        else:
            label[i, 0] = -1

    learning_rate = 0.01

    # Training loop for A、B
    for epoch in range(1000):
        z = dataPointSet.dot(A) + b

        # Compute the gradient for A
        # Compute the gradient for b
        sech_squared = 1 / np.cosh(z)**2
        gradient = dataPointSet.T.dot(sech_squared.dot(c))
        gradient_b = np.sum(sech_squared.dot(c), axis=0)

        # Update A
        # Update b
        A -= learning_rate * gradient
        b -= learning_rate * gradient_b

    # Now you can make predictions using the trained model
    z = dataPointSet.dot(A) + b
    loss = -z.dot(c)
    guess = np.sign(loss)  # Binary classification, assuming 1 and -1 are the two classes
    #print(guess)
    #print(label)
    err = np.sum(guess != label)  # Count the number of misclassified points
    
    if err<ans:
        ans=err
        correct_c=c
        
c=correct_c

random.seed(seed)
dataPointSe = np.random.rand(n0, m)  # Initialize dataPointSet with random values
dataPointSet = dataPointSe * 10 - 5
label = np.zeros((n0, 1))

for i in range(n0):
    if dataPointSet[i, 0]**2 + dataPointSet[i, 1]**2 >= 5:
        label[i, 0] = 1
    else:
        label[i, 0] = -1

learning_rate = 0.01

# Training loop for A、B
for epoch in range(10000):
    z = dataPointSet.dot(A) + b

    # Compute the gradient for A
    # Compute the gradient for b
    sech_squared = 1 / np.cosh(z)**2
    gradient = dataPointSet.T.dot(sech_squared.dot(c))
    gradient_b = np.sum(sech_squared.dot(c), axis=0)


    # Update A
    # Update b
    A -= learning_rate * gradient
    b -= learning_rate * gradient_b

z = dataPointSet.dot(A) + b
loss = -z.dot(c)
guess = np.sign(loss)  # Binary classification, assuming 1 and -1 are the two classes
#print(guess)
#print(label)
err = np.sum(guess != label)  # Count the number of misclassified points
print(err)
