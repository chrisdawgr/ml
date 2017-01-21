import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator
import math
from array import array

# Calculates the distance vector from a point to the rest of the points
# in the data. Sorts this list and gets the closest k neighbours
# then decides which target the k neighbours vote on.
def MSE(data, t):
  return 1 / float(2 * len(data)) * sum((t[i] - data[i]) ** 2\
    for i, d in enumerate(data))

def getKNNLoss(testData,setData,k):
  loss = 0
  (testFeatures,testTargets) = testData
  (trainFeatures,trainTargets) = setData
  """
  trainFeatures = np.loadtxt("ML2016GalaxiesTrain.dt1")
  trainTargets = np.loadtxt("ML2016SpectroscopicRedshiftsTrain.dt")
  testFeatures = np.loadtxt("ML2016GalaxiesTest.dt1")
  testTargets = np.loadtxt("ML2016SpectroscopicRedshiftsTest.dt")
  """
  yhat_l = []
  length = len(testFeatures)
  for i in range(len(testFeatures)):
    print str(i) + str( "/" ) + str(length)
    yhat = findKNN(testFeatures[i],trainFeatures,trainTargets,k)
    yhat_l.append(yhat)
  yhat_l = np.array(yhat_l)
  return MSE(testTargets,yhat_l)

# Takes an input feature a trainingset and targets for these datapoints
# computes the predictive valie i.e. the class attribute of the majority of
# the k-nearest neighbours
def findKNN(x,feats,targets,k):
  distanceArray = getDistanceMat(x,feats)
  sortedIndecies = distanceArray.argsort()
  classifiercount = {}
  regression = 0
  for i in range(k):
    votetarget = targets[sortedIndecies[i]]
    regression += votetarget
    #print regression

    classifiercount[votetarget] = classifiercount.get(votetarget,0)+1
  sortedclassifiercount = max(classifiercount.iteritems(),
    key=operator.itemgetter(1))[0]
  return regression/k #sortedclassifiercount

# Calculate the euclidean distance between two vectors
def getDistanceTwoPoints(x,xi):
  inner = 0
  for i in range(len(x)):
    inner += (x[i] - xi[i])**2
  return math.sqrt(inner)

# Calculate the distance vector for a point and a vector of points
def getDistanceMat(x,Mat):
  returnArray = np.zeros(len(Mat))
  for i in range(len(Mat)):
    returnArray[i] = getDistanceTwoPoints(x,Mat[i])
  return returnArray

# returns targetvalues and features of a dataset. Assumes that the features
# is at the last column

# takes a list of arrays and an index i. Removes the ith dataset and uses this
# for validation, merges the rest into the training data.
def removeRest(i,listofarrays):
  trainingset = []
  validation = listofarrays[i]
  for j in range(len(listofarrays)):
    if i != j:
      trainingset.append(listofarrays[j])
  comb= np.vstack([trainingset[0],trainingset[1],trainingset[2],trainingset[3]])
  return (comb,validation)

# shuffles the data randomly, split data into n partitions
def split(Data,splits):
  np.take(Data,np.random.permutation(Data.shape[0]),axis=0,out=Data)
  return np.split(Data,splits)

def prepareData_(Data):
  numFeatures = Data.shape[1]-1
  targetvalues = Data[:,numFeatures]
  features = np.delete(Data, numFeatures, axis=1)
  return(targetvalues,features)

# perform cross validation
def crossValidate(setData,splits,flag):
  (trainFeatures,trainTargets) = setData
  #print "her"
  #print trainFeatures.shape
  #trainTargets = trainTargets.reshape(5000,1)
  #Data = trainTargets
  #print "her2"
  #print trainTargets.shape
  #Data = Data.reshape(5000,1) #np.reshape(5000,1)
  data = np.hstack((trainFeatures, np.zeros((trainFeatures.shape[0], 1), dtype=trainFeatures.dtype)))
  #print data[:,-1].shape
  data[:,-1] = trainTargets
  kvals = [x for x in range(26) if x&1 != 0]
  #kvals = [27,29,31,33,35,37,39,41,43,45]
  missAvgList = []
  folds = split(data,splits)
  missAvgList = []
  kcountMisses = {}
  for k in kvals:
    kcountMisses[k] = 0
    for i in range(len(folds)):
      yhat_l = []
      print "fold: " + str(i)
      (trainingset,validation) = removeRest(i,folds)
      (Val_targetvalues, Val_features) = prepareData_(validation)
      (Train_targetvalues, Train_features) = prepareData_(trainingset)
      for j in range(len(Val_features)):
        yhat = findKNN(Val_features[j],Train_features,Train_targetvalues,k)
        #print yhat
        yhat_l.append(yhat)
      yhat_l = np.array(yhat_l)
      error = MSE(Val_targetvalues,yhat_l)
      kcountMisses[k] += error
    kcountMisses[k] = (kcountMisses[k] / float(splits))
    print "k: "+ str(k) + " error: " + str(error)
    missAvgList.append(kcountMisses[k])
  return missAvgList

def autoNorm(trainData,testData):
    (testdata,testval) = testData
    (dataSet,val) = trainData
    mean = np.sum(dataSet,axis=0)/len(dataSet)
    variance = (np.sum((dataSet - mean)**2,axis=0)/len(dataSet))
    meanTrain = mean; varTrain = variance
    std = np.sqrt(variance)
    normalizedTest = (testdata - mean) / np.sqrt(variance)
    normalizedData = (dataSet - mean) / np.sqrt(variance)
    meanTest = np.mean(normalizedTest,axis=0)
    varTest  = np.var(normalizedTest,axis=0)
    return normalizedTest,normalizedData,meanTrain,varTrain,meanTest,varTest

def plotcrossvalidation():
  kvals = [x for x in range(26) if x&1 != 0]

  p1, = plt.plot(kvals,miss,label='Average loss')
  plt.legend(handles=[p1],loc=1)
  plt.ylabel('Average loss over 5 folds')
  plt.xlabel('k')
  plt.show()
  return miss

def printCross(crossValres):
  kvals = [x for x in range(26) if x&1 != 0]
  for i in range(len(crossValres)):
    print "k = " + str(kvals[i]) + ": "  + str(crossValres[i])

# Main function for running assignment 1.1
def main():
  trainFeatures = np.loadtxt("ML2016GalaxiesTrain.dt1")
  trainTargets = np.loadtxt("ML2016SpectroscopicRedshiftsTrain.dt")

  testFeatures = np.loadtxt("ML2016GalaxiesTest.dt1")
  testTargets = np.loadtxt("ML2016SpectroscopicRedshiftsTest.dt")
  ks = [1,3,5]
  testData = (testFeatures,testTargets)
  trainData = (trainFeatures,trainTargets)
  #print trainFeatures.shape


  #trainingData = np.loadtxt("IrisTrainML.dt")
  #testData = np.loadtxt("IrisTestML.dt")
  print "Assignment 1.1: \n"
  """
  trainFeatures = np.loadtxt("ML2016GalaxiesTrain.dt1")
  trainTargets = np.loadtxt("ML2016SpectroscopicRedshiftsTrain.dt")

  testFeatures = np.loadtxt("ML2016GalaxiesTest.dt1")
  testTargets = np.loadtxt("ML2016SpectroscopicRedshiftsTest.dt")
  trainData = (trainFeatures,trainTargets)
  testData = (testFeatures,testTargets)
  """
  normalizedTest,normalizedData,meanTrain,varTrain,meanTest,varTest= autoNorm(trainData,testData)
  print normalizedData
  normTrain = (normalizedData,trainTargets)
  miss = crossValidate(normTrain,5,True)
  print miss
  #getKNNLoss(trainData,trainData,3)



  """
  for k in ks:
    loss = getKNNLoss(trainingData,trainingData,k)
    loss1 = getKNNLoss(testData,trainingData,k)
    print "Loss for k = " + str(k) + " Training data: " + str(loss)\
    + " Test data: " + str(loss1)
  print "\n"
  print "Assignment 1.2: \n"
  miss = crossValidate("IrisTrainML.dt",5,True)
  print "Result of performing crossvalidate:"
  printCross(miss)
  (dattest,dattrain,meanTrain,varTrain,meanTest,varTest) =\
    autoNorm("IrisTrainML.dt", "IrisTestML.dt")
  print "\n"
  print "Assignment 1.3 \n"
  print "Mean of training data: " + str(meanTrain)
  print "Variance of training data " + str(varTrain)
  print "Mean of test data " + str(meanTest)
  print "variance of test data " + str(varTest)
  """
main()