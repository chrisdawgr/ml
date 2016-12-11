import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import kmeans as km

filename = "ML2016TrafficSignsTrain.csv"

"""
Load the data and remove targetValues
"""
def prepareData(filename):
  data = np.loadtxt(open(filename,"rb"),delimiter=",")
  numberOfPixels = len(data[0])-1
  features = np.delete(data, numberOfPixels, axis=1)
  targetValues = data[:,numberOfPixels]
  return (features,targetValues)

"""
Transform datalabels into shapes of trafficsigns
"""
def assignLabels(target):
  shapevalues = []
  for i in range(len(target)):
    if (target[i] < 11) or (target[i] > 31)\
      or (target[i] == 15) or (target[i] == 16) or (target[i] ==17):
      shapevalues.append(0)
    elif target[i] == 12:
      shapevalues.append(2)
    elif target[i] == 13:
      shapevalues.append(3)
    elif target[i] == 14:
      shapevalues.append(4)
    elif target[i] == 11 or (target[i] > 18 or target[i] < 32):
      shapevalues.append(1)
  return shapevalues

"""
Compute the k principle components
"""
def getPCA(filename,k):
  features,targetValues = prepareData(filename)
  shapes_signs = assignLabels(targetValues)
  meanVals = np.mean(features, axis=0)
  meanRemoved = features - meanVals
  covMat = np.cov(meanRemoved,rowvar=0)
  eigVals,eigVects = np.linalg.eig(np.mat(covMat))
  eigVals = np.real(eigVals)
  eigVects = np.real(eigVects)
  eigValInd = np.argsort(eigVals)
  topvalList = eigVals[eigValInd][::-1]
  # To plot the eigenspectrum:
  #plotSpectrum(topvalList)
  eigValInd = eigValInd[:-(k+1):-1]
  sortFeat = targetValues[eigValInd]
  redEigVects = eigVects[:,eigValInd]
  percentile = 0
  eigenSum = np.sum(eigVals)
  for i in range(len(topvalList)):
    percentile += np.real(topvalList[i])
    percent = percentile / eigenSum
    if percentile/eigenSum > 0.90:
      component = i
      break
  z = np.dot(meanRemoved,redEigVects)
  x =  z[:,0]
  y =  z[:,1]

  plot(x,y,shapes_signs,meanRemoved,redEigVects)

"""
Function for plotting the projected data and the 4 clusters
"""
def plot(x,y,label,meanRemoved,redEigVects):
  for i in range(1):
    print "plot iteration: " + str(i)
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    c = ['red','green','blue','orange','black','yellow']
    m = ['o','^','D','v','8']
    al = [0.45,1,1,1,1,1]
    for j in range(len(x)):
      ax.scatter(x[j],y[j],c=c[label[j]],marker=m[label[j]],alpha=al[label[j]])
    clusters = km.kMeans(meanRemoved,4)
    reducedClusters = np.dot(clusters,redEigVects)
    x1 = reducedClusters[:,0]
    y1 = reducedClusters[:,1]
    cluster = ax.scatter(x1,y1,label='Clusters',c='green',s=700,marker='.')
    plt.legend(handles=[cluster],loc=3)
    plt.savefig(str(i) + ' figure')
    plt.close()
    #plt.show()

"""
Function for plotting the eigenspectrum
"""
def plotSpectrum(yvals):
  xvals = [x for x in range(len(yvals))]
  p1, = plt.plot(xvals,yvals,label='Average loss')
  plt.title("Eigenspectrum")
  plt.yscale('log')
  plt.xlabel('Principle Components')
  plt.ylabel('Eigenvalues')
  plt.show()

getPCA(filename,2)