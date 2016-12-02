import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator
import math
from array import array


# Assignment 1.1
# report zero one loss on training and test data for k=1,2,3

# Calculates the distance vector from a point to the rest of the points
# in the data. Sorts this list and gets the closest k neighbours
# then decides which target the k neighbours vote on.

def getKNNLoss(testData,setData,k):
  print "calling getknnloss:" + str(k)
  loss = 0
  testTargets,testFeatures = prepareData_(testData)
  setTargets,setFeatures = prepareData_(setData)
  for i in range(len(testData)):
    yhat = findKNN(testFeatures[i],setFeatures,setTargets,k)
    if yhat != testTargets[i]:
      loss += 1
  return (float(loss)/len(testData))

def findKNN(x,feats,targets,k):
  distanceArray = getDistanceMat(x,feats)
  sortedIndecies = distanceArray.argsort()
  classifiercount = {}
  for i in range(k):
    votetarget = targets[sortedIndecies[i]]
    classifiercount[votetarget] = classifiercount.get(votetarget,0)+1
  sortedclassifiercount = max(classifiercount.iteritems(),
    key=operator.itemgetter(1))[0]
  return sortedclassifiercount#[0][0]

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

def prepareData_(Data):
  numFeatures = Data.shape[1]-1
  targetvalues = Data[:,numFeatures]
  features = np.delete(Data, numFeatures, axis=1)
  return(targetvalues,features)

def main(trainingFile,testFile):
  ks = [1,3,5]
  trainingData = np.loadtxt(trainingFile)
  testData = np.loadtxt(testFile)

  #ks = [3]
  for k in ks:
    #k = ks[i]

    loss = getKNNLoss(trainingData,trainingData,k)
    loss1 = getKNNLoss(testData,trainingData,k)
    #loss2 = getKNNLoss(trainNorm,trainNorm,k)
    #loss3 = getKNNLoss(testNorm,trainNorm,k)
    #loss  = computeKNN("IrisTrainML.dt",k)
    #loss1 = testcaseKNN("IrisTestML.dt","IrisTrainML.dt",k)
    print "Loss for k = " + str(k) + " Training data: " + str(loss)\
    + " Test data: " + str(loss1)
    #print "Normalized data:"
    #print "Loss for k = " + str(k) + " Training data: " + str(loss2)\
    #+ " Test data: " + str(loss3)


# validation is only trainingset without target values i.e the ones which
# should be estimated for and the ones where we want to test. trainingset
# is other values to be tested against

# run

#[array1 array2 array3 array4 array 5]
# TODO should be validation 1 rest should be merged into one test
def removeRest(i,listofarrays):
  #print listofarrays
  trainingset = []
  validation = listofarrays[i]
  for j in range(len(listofarrays)):
    if i != j:
      trainingset.append(listofarrays[j])
  comb= np.vstack([trainingset[0],trainingset[1],trainingset[2],trainingset[3]])
  return (comb,validation)

def split(Data,splits):
  #NOTE: assumes array is divisable by number of splits
  #np.take(Data,np.random.permutation(Data.shape[0]),axis=0,out=Data)
  return np.split(Data,splits)

# NOTE: Save number of loses for each k (to find best value of k)
#     :

def savetotext(filename, array):
  text_file = open(filename, "w")
  for i in range(len(array)):
    text_file.write(str(array[i]))
    text_file.write("\n")
  text_file.close()

def crossValidate(filename,splits,flag):
  kvals = [x for x in range(26) if x&1 != 0]
  missAvgList = []
  if flag == True:
    Data = np.loadtxt(filename)
    #print Data
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
          #print "miss! k: " + str(k) + "fold: " + str(i) + " target: " + str(Val_features[j])
          kcountMisses[k] += 1
    kcountMisses[k] = (kcountMisses[k] / float(splits)) / len(Val_features)
    missAvgList.append(kcountMisses[k])
  return missAvgList

#miss = crossValidate("IrisTrainML.dt",5)
#rint miss

# 1.3 data normalization
# zero mean, remove mean from every point
# unit variance divide by standert diviation
def autoNorm(training,test):
    Data = np.loadtxt(training)
    test = np.loadtxt(test)
    (testval,testdata) = prepareData_(test)
    (val,dataSet) = prepareData_(Data)
    mean = np.sum(dataSet,axis=0)/len(Data)
    #mean1 = np.mean(dataSet,axis=0)
    print "Mean: " + str(mean)
    variance = (np.sum((dataSet - mean)**2,axis=0)/len(Data))
    print "Variance: " + str(variance)
    std = np.sqrt(variance)
    #print dataSet
    #print mean
    #print dataSet-mean
    normalizedTest = (testdata - mean) / np.sqrt(variance)
    normalizedData = (dataSet - mean) / np.sqrt(variance)
    return normalizedTest,normalizedData

test,data = autoNorm("IrisTrainML.dt","IrisTestML.dt")
print np.mean(test,axis=0)
print np.var(test,axis=0)
#print "calc variance: " +  str(np.var(data[:,0])) + " " + str(np.var(data[:,1]))
#print "calc mean: " + str(np.mean(data[:,0])) + " " + str(np.mean(data[:,1]))
#print data

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


#plot("IrisTrainML.dt",False)
#plot("IrisTrainML.dt",True)



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


  #print Data
  return miss

def plotcrossvalidation():
  kvals = [x for x in range(26) if x&1 != 0]
  #miss = crossValidate("IrisTrainML.dt",5,True)
  miss = getNormCrossValidation()
  p1, = plt.plot(kvals,miss,label='Average loss')
  plt.legend(handles=[p1],loc=1)
  plt.ylabel('Average loss over 5 folds')
  plt.xlabel('k')
  plt.show()
  return miss

miss = plotcrossvalidation()
print miss

#miss = getNormCrossValidation()
#print miss

#miss = crossValidate("IrisTrainML.dt",5,True)
#print miss

main("IrisTrainML.dt","IrisTestML.dt")


#plotcrossvalidation()