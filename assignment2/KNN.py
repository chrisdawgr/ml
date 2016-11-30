import numpy as np
import operator
import math

# Assignment 1.1
# report zero one loss on training and test data for k=1,2,3

def computeKNN(filename,k):
  # compares for each point in the data the classified target, and the actual
  # target. If the entry has been classified wrongly the loss is incremented.
  Data = np.loadtxt(filename)
  targetvalues,features = prepareData_(Data)
  loss = 0
  for i in (range(len(targetvalues))):
    # NOTE : remove or not remove?
    # NOTE: prev feats,targets = removeFromData(i,features,targetvalues)
    yhat = findKNN(i,features[i],features,targetvalues,k)
    if yhat != targetvalues[i]:
      loss += 1
  return (float(loss)/len(Data))

# Calculates the distance vector from a point to the rest of the points
# in the data. Sorts this list and gets the closest k neighbours
# then decides which target the k neighbours vote on.
def findKNN(i,x,feats,targets,k):
  distanceArray = getDistanceMat(x,feats)
  sortedIndecies = distanceArray.argsort()
  classifiercount = {}
  for i in range(k):
    votetarget = targets[sortedIndecies[i]]
    classifiercount[votetarget] = classifiercount.get(votetarget,0)+1
  sortedclassifiercount = sorted(classifiercount.iteritems(),
                            key = operator.itemgetter(1))
  return sortedclassifiercount[0][0]

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

# Remove a given indencie from a set of features and targets
def removeFromData(i,features,targetvalues):
  feats   = np.delete(features,i,axis=0)
  targets = np.delete(targetvalues,i,axis=0)
  return (feats,targets)

def prepareData_(Data):
  numFeatures = Data.shape[1]-1
  targetvalues = Data[:,numFeatures]
  features = np.delete(Data, numFeatures, axis=1)
  return(targetvalues,features)

def main():
  ks = [1,3,5]
  for i in range(3):
    k = ks[i]
    loss  = computeKNN("IrisTrainML.dt",k)
    loss1 = computeKNN("IrisTestML.dt",k)
    print "Loss for k = " + str(k) + " Training data: " + str(loss)\
    + " Test data: " + str(loss1)

main()

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

def split(filename,splits):
  Data = np.loadtxt(filename)
  #NOTE: assumes array is divisable by number of splits
  np.take(Data,np.random.permutation(Data.shape[0]),axis=0,out=Data)
  return np.split(Data,splits)

# NOTE: Save number of loses for each k (to find best value of k)
#     :
def crossValidate(filename,splits):
  kvals = [x for x in range(26) if x&1 != 0]
  #kvals = [1]
  folds = split(filename,splits)
  # for each k
  kcountMisses = {}
  for k in kvals:
    kcountMisses[k] = 0
    # perform 5 folds
    for i in range(len(folds)):
      (trainingset,validation) = removeRest(i,folds) #incorrect
      (Val_targetvalues, Val_features) = prepareData_(validation) #correct
      (Train_targetvalues, Train_features) = prepareData_(trainingset) # correct
      # for each entry item in the validatin set
      for j in range(len(Val_features)):
        yhat = findKNN(j,Val_features[j],Train_features,Train_targetvalues,k)
        if yhat != Val_targetvalues[j]:
          kcountMisses[k] += 1
    kcountMisses[k] = kcountMisses[k] / float(splits)
  return kcountMisses



miss = crossValidate("IrisTrainML.dt",5)
print miss

# 1.3 data normalization
# zero mean, remove mean from every point
# unit variance divide by standert diviation
def autoNorm(dataSet):
    # get biggest and smallest values of x
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
