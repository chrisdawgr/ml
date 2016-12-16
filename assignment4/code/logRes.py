from __future__ import division
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

filename = "IrisTrainML.dt"

def prepareData(filename):
  data = np.loadtxt(open(filename,"rb"),delimiter=" ")
  numberOfPixels = len(data[0])-1
  removedData = []
  for i in range(len(data)):
    if data[i][2] != 2:
      removedData.append(data[i])
  data = np.array(removedData)
  features = np.delete(data, numberOfPixels, axis=1)
  newData = np.ones([len(features),1])
  features = np.concatenate((newData,features),axis=1)
  targetvalues = data[:,2]
  for i in range(len(targetvalues)):
    if targetvalues[i] == 0:
      targetvalues[i] = -1
  return features,targetvalues

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradDesc(dataMatIn,classLabels):
  dataMatrix = np.mat(dataMatIn)
  alpha = 0.1
  maxCycles = 10000
  labelMat = np.mat(classLabels).transpose()
  m,n = np.shape(dataMatrix)
  weights = np.zeros((n,1))
  for k in range(maxCycles):
    grad = 0
    for n in range(len(dataMatrix)):
      grad += (classLabels[n] * dataMatrix[n])/ \
      (1 + math.exp(classLabels[n].transpose() * weights.transpose() * dataMatrix[n].transpose()))
    gradient = ((-1/len(dataMatrix))*grad).transpose()
    weights = weights + alpha * -gradient
  return weights

def plot(features,label):
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  colors = ['red','green']
  ax.scatter(features[:,0],features[:,1],c=label,
    cmap=matplotlib.colors.ListedColormap(colors))
  plt.show()

def main():
  features, targetvalues = prepareData(filename)
  weights = gradDesc(features,targetvalues)
  print "Weights are: " + str(weights)
  regressed = (weights.transpose() * features.transpose())
  regressed = np.array(regressed)
  error = 0
  for i in range(len(features)):
    if sigmoid(regressed[0][i]) > 0.5:
      predicted = 1
    else:
      predicted = -1
    if predicted != targetvalues[i]:
      error += 1
  errorTrain = error/len(features)
  print "iterations: 10000"
  print "step size : 0.01"
  print "Emperical training error: " + str(errorTrain)
  featuresTest, targetvaluesTest = prepareData("IrisTestML.dt")
  regressedTest = (weights.transpose() * featuresTest.transpose())
  regressedTest = np.array(regressedTest)
  errorTest = 0
  for i in range(len(featuresTest)):
    if sigmoid(regressedTest[0][i]) > 0.5:
      predictedTest = 1
    else:
      predictedTest = -1
    if predictedTest != targetvaluesTest[i]:
      errorTest += 1
  errorTest = errorTest/len(featuresTest)
  print "iterations: 10000"
  print "step size : 0.01"
  print "Emperical training error: " + str(errorTest)

main()