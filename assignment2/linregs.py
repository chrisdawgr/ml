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

def standRegres(xArr,yArr):
  xMat = np.mat(xArr); yMat = np.mat(yArr).T
  xTx = xMat.T*xMat
  if np.linalg.det(xTx) == 0.0:
    print "This matrix is singular, cannot do inverse"
    return
  ws = xTx.I * (xMat.T*yMat)
  return ws

def linRegs(filename):
  Data = np.loadtxt(filename)
  (Y,X) = prepareData_(Data)
  newData = np.ones([len(X),1])
  X =  X.reshape((6, 1))
  X = np.concatenate((newData,X),axis=1)
  Y = Y.reshape(1,6)
  weights = standRegres(X,Y)
  error = MSE(X,Y,weights)
  print error
  return (weights,error)

def MSE(X,Y,weights):
  error = 0
  numdata = len(X)
  yHat = X*weights
  for i in range(numdata):
    error += (yHat[i]-Y[0][i])**2
  return error / numdata

def plot(filename):
  Data = np.loadtxt(filename)
  fig = plt.figure()
  ax  = fig.add_subplot(111)
  (weights,error) = linRegs(filename)
  print weights
  x = Data[:,0]
  y = Data[:,1]
  x = x.reshape((6, 1))
  y = y.reshape((6,1))
  p1, = plt.plot(x, y, '.',label='Data')
  p2, = plt.plot(x, (1*weights[0] + x* weights[1]), '-',label='Linear Regression')
  plt.ylabel('y')
  plt.xlabel('x')
  plt.legend(handles=[p1, p2],loc=2)
  plt.show()

plot("DanWood.dt")


#linRegs("DanWood.dt"