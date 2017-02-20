import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import operator
import math
from array import array

def prepareData_(Data):
  numFeatures = Data.shape[1]-1
  targetvalues = Data[:,numFeatures]
  features = np.delete(Data, numFeatures, axis=1)
  return(targetvalues,features)

# calculates the weights of the linear regression
def standRegres(xArr,yArr):
  xMat = np.mat(xArr); yMat = np.mat(yArr).T
  xTx = xMat.T*xMat
  if np.linalg.det(xTx) == 0.0:
    print "This matrix is singular, cannot do inverse"
    return
  ws = xTx.I * (xMat.T*yMat)
  return ws

# takes a filename and returns the weights and the RMSE of the dataset
def linRegs(filename):
  Data = np.loadtxt(filename)
  print Data
  (Y,X) = prepareData_(Data)
  newData = np.ones([len(X),1])
  X =  X.reshape((len(X), 1))
  X = np.concatenate((newData,X),axis=1)
  Y = Y.reshape(1,len(Y))
  weights = standRegres(X,Y)


  #1 bias (aka offset or intercept) parameter:




  error = MSE(X,Y,weights)
  print error
  return (weights,error)

# calculates the root mean squared error of a dataset and a linear model
def MSE(X,Y,weights):
  error = 0
  numdata = len(X)
  yHat = X*weights
  for i in range(numdata):
    error += (yHat[i]-Y[0][i])**2
  return error / numdata

# plots the linear regression of a dataset
def plot(filename):
  Data = np.loadtxt(filename)
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  (weights,error) = linRegs(filename)
  x = Data[:,0]
  y = Data[:,1]
  x = x.reshape((6, 1))
  y = y.reshape((6,1))
  p1, = plt.plot(x, y, '.',label='Data')
  p2, = plt.plot(x, (1*weights[0] + x* weights[1]),\
    '-',label='Linear Regression')
  plt.ylabel('y')
  plt.xlabel('x')
  plt.legend(handles=[p1, p2],loc=2)
  plt.show()

def main():
  weights,error = linRegs("DanWood.dt")
  print "Assignment 5: \n"
  print "Weights = " + str(weights[0]) + " " + str(weights[1])
  print "RMSE error on whole dataset = " + str(error)

main()