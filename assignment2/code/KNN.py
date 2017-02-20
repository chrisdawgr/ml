import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator
import math
from array import array

# Calculates the distance vector from a point to the rest of the points
# in the data. Sorts this list and gets the closest k neighbours
# then decides which target the k neighbours vote on.
def getKNNLoss(testData,setData,k):
  loss = 0
  testTargets,testFeatures = prepareData_(testData)
  setTargets,setFeatures = prepareData_(setData)
  for i in range(len(testData)):
    yhat = findKNN(testFeatures[i],setFeatures,setTargets,k)
    if yhat != testTargets[i]:
      loss += 1
  return (float(loss)/len(testData))

# Takes an input feature a trainingset and targets for these datapoints
# computes the predictive valie i.e. the class attribute of the majority of
# the k-nearest neighbours
def findKNN(x,feats,targets,k):
  distanceArray = getDistanceMat(x,feats)
  sortedIndecies = distanceArray.argsort()
  classifiercount = {}
  for i in range(k):
    votetarget = targets[sortedIndecies[i]]
    classifiercount[votetarget] = classifiercount.get(votetarget,0)+1
  sortedclassifiercount = max(classifiercount.iteritems(),
    key=operator.itemgetter(1))[0]
  return sortedclassifiercount

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
def prepareData_(Data):
  numFeatures = Data.shape[1]-1
  targetvalues = Data[:,numFeatures]
  features = np.delete(Data, numFeatures, axis=1)
  return(targetvalues,features)

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

# perform cross validation
def crossValidate(filename,splits,flag):
  kvals = [x for x in range(26) if x&1 != 0]
  missAvgList = []
  if flag == True:
    Data = np.loadtxt(filename)
  else:
    Data = filename
  folds = split(Data,splits)
  kcountMisses = {}
  for k in kvals:
    kcountMisses[k] = 0
    for i in range(len(folds)):
      (trainingset,validation) = removeRest(i,folds)
      (Val_targetvalues, Val_features) = prepareData_(validation)
      (Train_targetvalues, Train_features) = prepareData_(trainingset)
      for j in range(len(Val_features)):
        yhat = findKNN(Val_features[j],Train_features,Train_targetvalues,k)
        if yhat != Val_targetvalues[j]:
          kcountMisses[k] += 1
    kcountMisses[k] = (kcountMisses[k] / float(splits)) / len(Val_features)
    missAvgList.append(kcountMisses[k])
  return missAvgList

# perform normalization of test set with respect to values of trainingset
def autoNorm(training,test):
    Data = np.loadtxt(training)
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
    return normalizedTest,normalizedData,meanTrain,varTrain,meanTest,varTest

# plot dataset
def plot(filename,norm):
  Data = np.loadtxt(filename)
  label = Data[:,2]
  if norm == True:
    (Data) = autoNorm(filename)
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  colors = ['red','green','blue']
  ax.scatter(Data[:,0],Data[:,1],c=label,
    cmap=matplotlib.colors.ListedColormap(colors))
  plt.show()

# perform cross validation on normalized data
def getNormCrossValidation():
  kvals = [x for x in range(26) if x&1 != 0]
  Dat = np.loadtxt("IrisTrainML.dt")
  numFeatures = Dat.shape[1]-1
  targetvalues = Dat[:,numFeatures]
  jib,Data = autoNorm("IrisTrainML.dt","IrisTrainML.dt")
  print Data.shape
  print targetvalues.shape
  targetvalues = targetvalues.reshape(100,1)
  data = np.append(Data,targetvalues,axis=1)
  miss = crossValidate(data,5,False)
  return miss

def plotcrossvalidation():
  kvals = [x for x in range(26) if x&1 != 0]
  miss = getNormCrossValidation()
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
  ks = [1,3,5]
  trainingData = np.loadtxt("IrisTrainML.dt")
  testData = np.loadtxt("IrisTestML.dt")
  print "Assignment 1.1: \n"
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

main()