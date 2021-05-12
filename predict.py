import numpy as np
import csv
import sys
import matplotlib.pyplot as plt

def addColsOfOnesAtStart(X):
    temp = np.ones((X.shape[0], X.shape[1]+1))
    temp[:, 1:] = X
    return temp 

def computeHypothesis(X,W):
    return np.dot(X,W)

def computeCost(X,W,Y): 
    return np.sum((computeHypothesis(X,W) - Y)**2)/(2*X.shape[0])

def computeGradients(X,W,Y):
    return (np.dot(np.transpose(X),(computeHypothesis(X,W) - Y))/X.shape[0])

if len(sys.argv) < 2:
    print("Missing test file path!")
    exit(0)
test_X_filepath = sys.argv[1]
X_test = []
with open(test_X_filepath, "r") as fp: 
    lines = fp.readlines()
    lines = lines[1:]
    for line in lines: 
        X_test.append([float(x) for x in line.split(",")])
X_test = np.array(X_test)
X_test = addColsOfOnesAtStart(X_test)

X, Y = [], []
with open("train_X_lr.csv", "r") as fp: 
    lines = fp.readlines()
    lines = lines[1:]
    for line in lines: 
        X.append([float(x) for x in line.split(",")])
with open("train_Y_lr.csv", "r") as fp: 
    lines = fp.readlines()
    for line in lines: 
        Y.append(float(line))
X = np.array(X)
Y = np.array(Y)
Y = Y.reshape(Y.shape[0],1)
X = addColsOfOnesAtStart(X)

X_val = X[250:, :]
Y_val = Y[250:, :]
X = X[:250, :]
Y = Y[:250, :]

# W = np.random.standard_normal(size=(X.shape[1], 1))
W = np.zeros((X.shape[1], 1))
# print(f"X shape = {X.shape}")
# print(f"Y shape = {Y.shape}")
# print(f"W shape = {W.shape}")

#Momentum based GD
numEpochs = 50000
learningRate = 0.0003
v = np.zeros(W.shape)
gamma = 0.9996
costs = []
valCosts = []
for i in range(numEpochs):
    Ypred = computeHypothesis(X,W)
    diffs = (Ypred - Y)
    gradW = np.dot(np.transpose(X), diffs)
    v = v*gamma + (learningRate) * (gradW/X.shape[0])
    W -= v
    cost = (np.sum((diffs**2)))/(2*X.shape[0])
    costs.append(cost)
    valCost = (np.sum(((np.dot(X_val,W) - Y_val)**2)))/(2*X_val.shape[0])
    valCosts.append(valCost)
    print(f"Epoch {i+1}: Cost: {cost}, ValCost: {valCost}")
with open("predicted_test_Y_lr.csv","w") as fp: 
    for i in range(X_test.shape[0]):
        fp.write(str(computeHypothesis(X_test[i],W)[0]) + "\n")

# plt.plot(list(range(numEpochs)), costs)
# plt.plot(list(range(numEpochs)), valCosts, color="red")
# plt.show()
