from __future__ import division
import numpy as np
import math


import sys

def remPunc(line):
  outputLine = ""
  for letter in line:
    if letter == ",":
      outputLine += " "
    else:
      outputLine += letter
  return outputLine

def parseText(inputfile):
  openfile = open(inputfile,'r')
  outputfile = inputfile + "1"
  writefile = open(outputfile,'w')
  for line in openfile:
    line = remPunc(line)
    writefile.write(line)
  writefile.flush()
  writefile.close()
  openfile.close()

def prepareData(Data):
  numFeatures = Data.shape[1]-1
  targetvalues = Data[:,numFeatures]
  features = np.delete(Data, numFeatures, axis=1)
  for i in range(len(targetvalues)):
    if targetvalues[i] == 0:
      targetvalues[i] = -1
  return(features,targetvalues)

parseText('ML2016WeedCropTest.csv')
parseText('ML2016WeedCropTrain.csv')

test = np.loadtxt("ML2016WeedCropTest.csv1")
train = np.loadtxt("ML2016WeedCropTrain.csv1")

(train_X,train_Y) = prepareData(train)
(test_X,test_Y) = prepareData(test)

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradDesc(dataMatIn,classLabels):
  dataMatrix = np.mat(dataMatIn)
  alpha = 0.00001
  maxCycles = 5000
  labelMat = np.mat(classLabels).transpose()
  m,n = np.shape(dataMatrix)
  weights = (np.random.random_sample([n,1])/100)
  for k in range(maxCycles):
    grad = 0
    print str(k) + " / " + str(maxCycles)
    for n in range(len(dataMatrix)):
      grad += (classLabels[n] * dataMatrix[n])/ \
      (1 + math.exp(classLabels[n].transpose() * weights.transpose() * dataMatrix[n].transpose()))
    gradient = ((-1/len(dataMatrix))*grad).transpose()
    weights = weights + alpha * -gradient
  return weights

def main():
  features,targetvalues = train_X, train_Y
  #features, targetvalues = prepareData(filename)
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
  errorTrain = error#/len(features)
  print "iterations: 10000"
  print "step size : 0.01"
  print "Emperical training error: " + str(errorTrain)
  featuresTest, targetvaluesTest = test_X,test_Y #prepareData("IrisTestML.dt")

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
  errorTest = errorTest#/len(featuresTest)
  print "iterations: 10000"
  print "step size : 0.01"
  print "Emperical test error: " + str(errorTest)

main()
