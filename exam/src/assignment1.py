import numpy as np
import string
import re

# vaariance, calculate the mean, take squrae of differenceaaa

#filenames


def remPunc(line):
  outputLine = ""
  for letter in line:
    #if letter == ".":
    #  outputLine += ","
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

"""
parseText('ML2016GalaxiesTrain.dt')
parseText('ML2016GalaxiesTest.dt')
"""
trainFeatures = "ML2016GalaxiesTrain.dt1"
trainTargets  = "ML2016SpectroscopicRedshiftsTrain.dt"
testFeatures = "ML2016GalaxiesTest.dt1"
testTargets  = "ML2016SpectroscopicRedshiftsTest.dt"

weights1 = [-0.801881,0.0185134,0.0479647,-0.0210943,-0.0274002,-0.0226798,0.0064449,0.0151842,0.0120738,0.0103486,0.00599684,-0.0294513,0.069059,0.00630583,-0.00472042,-0.00873932,0.00311043,0.0017252,0.00435176]


def get_variance(data):
  Data = np.loadtxt("ML2016SpectroscopicRedshiftsTrain.dt")
  return np.sum(np.power((Data-(np.sum(Data)/float(len(Data)))),2)) /len(Data)


def MSE(data,target):
  data = np.loadtxt(data)
  target = np.loadtxt(target)
  sqDiff = np.power(data-target,2)
  sumDiff = np.sum(sqDiff)
  return (1.0 / len(data)) * sumDiff

def prepareData_(Data,Target):
  numFeatures = Data.shape[1]
  #targetvalues = Data[:,numFeatures]
  features = np.delete(Data, numFeatures, axis=1)
  print features.shape
  return(targetvalues,features,numFeatures)

# linregs on ML2016GalaxiesTrain.dt and ML2016SpectroscopicRedshiftsTrain.dt
# evaluate on ML2016GalaxiesTest.dt and ML2016SpectroscopicRedshiftsTest.dt

def MSE1(X,Y,weights):
  error = 0
  numdata = len(X)
  yHat = X*weights
  for i in range(numdata):
    error += (yHat[i]-Y[0][i])**2
  return error / numdata

def prepareData(X,Y):
  numFeats = X.shape[1]
  newData = np.ones([len(X),1])
  X =  X.reshape((len(X), numFeats))
  X = np.concatenate((newData,X),axis=1)
  Y = Y.reshape(1,len(Y))
  return X,Y

def getError():
  parseText('ML2016GalaxiesTrain.dt')
  parseText('ML2016GalaxiesTest.dt')

  X_train = np.loadtxt(trainFeatures)
  Y_train = np.loadtxt(trainTargets)
  X_test = np.loadtxt(testFeatures)
  Y_test = np.loadtxt(testTargets)

  X_train,Y_train = prepareData(X_train,Y_train)
  X_test,Y_test = prepareData(X_test,Y_test)
  weights1 = [-0.801881,0.0185134,0.0479647,-0.0210943,-0.0274002,-0.0226798,0.0064449,0.0151842,0.0120738,0.0103486,0.00599684,-0.0294513,0.069059,0.00630583,-0.00472042,-0.00873932,0.00311043,0.0017252,0.00435176]
  weights1 = np.array(weights1)
  weights1 = np.mat(weights1.reshape(19,1))
  trainError = MSE1(X_train,Y_train,weights1)
  testError = MSE1(X_test,Y_test,weights1)
  return trainError,testError


def main():
  var =\
  get_variance("ML2016SpectroscopicRedshiftsTrain.dt")
  mse =\
  MSE("ML2016EstimatedRedshiftsTest.dt","ML2016SpectroscopicRedshiftsTest.dt")
  print "variance: " + str(var)
  print "MSE: " + str(mse)
  trainError, testError = getError()
  print "Training error: " + str(trainError)
  print "Test error: " + str(testError)

main()
