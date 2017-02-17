from __future__ import division
import numpy as np
import math


import sys

def prepareData_(Data):
  numFeatures = Data.shape[1]-1
  targetvalues = Data[:,numFeatures]
  for i in range(len(targetvalues)):
    if targetvalues[i] == 0:
      targetvalues[i] = -1
  features = np.delete(Data, numFeatures, axis=1)
  return(targetvalues,features)

def Norm(train,test):
  Data = np.loadtxt(train)
  test = np.loadtxt(test)
  (testval,testdata) = prepareData_(test)
  (val,dataSet) = prepareData_(Data)
  mean = np.sum(dataSet,axis=0)/len(Data)
  variance = (np.sum((dataSet - mean)**2,axis=0)/len(Data))
  meanTrain = mean; varTrain = variance
  std = np.sqrt(variance)
  normalizedTest = (testdata - mean) / np.sqrt(variance)
  normalizedData = (dataSet - mean) / np.sqrt(variance)
  meanTest = np.mean(normalizedTest,axis=0)
  varTest  = np.var(normalizedTest,axis=0)
  return normalizedTest,testval,normalizedData,val

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

#parseText('ML2016WeedCropTest.csv')
#parseText('ML2016WeedCropTrain.csv')

test = np.loadtxt("ML2016WeedCropTest.csv1")
train = np.loadtxt("ML2016WeedCropTrain.csv1")

(train_X,train_Y) = prepareData(train)
(test_X,test_Y) = prepareData(test)

def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

def gradDesc(dataMatIn,classLabels,maxCycles,alpha):
  dataMatrix = np.mat(dataMatIn)
  #alpha = 0.00001
  #maxCycles = 2000
  labelMat = np.mat(classLabels).transpose()
  m,n = np.shape(dataMatrix)
  weights = np.zeros([n,1])
  #weights = (np.random.random_sample([n,1])/100)
  for k in range(maxCycles):
    grad = 0
    #print str(k) + " / " + str(maxCycles)
    for n in range(len(dataMatrix)):
      grad += (classLabels[n] * dataMatrix[n])/ \
      (1 + math.exp(classLabels[n].transpose() * weights.transpose() * dataMatrix[n].transpose()))
    gradient = ((-1/len(dataMatrix))*grad).transpose()
    weights = weights + alpha * -gradient
  return weights

def main1(norm,iterations,eps):
  features,targetvalues = train_X, train_Y
  featuresTest, targetvaluesTest = test_X,test_Y #prepareData("IrisTestML.dt")
  #print len(features)
  #print len(featuresTest)
  if norm:
    print "Running logistic regression on normalized data"
    (featuresTest,targetvaluesTest,features,targetvalues) =\
    Norm("ML2016WeedCropTrain.csv1","ML2016WeedCropTest.csv1")
  else:
    print "Running logistic regression on non-normalized data"
  #features, targetvalues = prepareData(filename)
  weights = gradDesc(features,targetvalues,iterations,eps)
  #weights = np.zeros
  #print "Weights are: " + str(weights)
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
  errorTrain = error
  print "iterations: " + str(iterations)
  print "step size : " + str(eps)
  print "Emperical training error: " + str(errorTrain)
  avg = errorTrain / float(len(features))
  print "average: " + str(avg)

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
  avg1 = errorTest/float(len(featuresTest))
  print "iterations: " + str(iterations)
  print "step size : " + str(eps)
  print "Emperical test error: " + str(errorTest)
  print "average: " + str(avg1)

def main():
  main1(False,2000,0.00001)
  main1(True,2000,0.0001)