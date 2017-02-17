import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.lines as mlines
import kmean as km

"""
Load the data and remove targetValues
"""
def prepareData(filename):
  #data = np.loadtxt(open(filename,"rb"),delimiter=",")
  data = np.loadtxt(filename)
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
  # seperate features and targetvalues
  features,targetValues = prepareData(filename)
  #print targetValues
  # determine the shapes of each signs
  shapes_signs = targetValues #assignLabels(targetValues)
  # calculate the mean of the input data
  meanVals = np.mean(features, axis=0)
  meanRemoved = features - meanVals
  # compute the covarinace matrix of the meanless data
  covMat = np.cov(meanRemoved,rowvar=0)
  # get teh eigenvalues and eigenvectors
  eigVals,eigVects = np.linalg.eig(np.mat(covMat))
  eigVals = np.real(eigVals); eigVects = np.real(eigVects)
  # sort the eigenvalues
  eigValInd = np.argsort(eigVals)
  topvalList = eigVals[eigValInd][::-1]
  # plot eigenspectrum
  plotSpectrum(topvalList)
  eigValInd = eigValInd[:-(k+1):-1]
  sortFeat = targetValues[eigValInd]
  # get principle components by retreiving the top k eigenvectors
  principleComponents = eigVects[:,eigValInd]
  percentile = 0
  eigenSum = np.sum(eigVals)
  for i in range(len(topvalList)):
    percentile += np.real(topvalList[i])
    percent = percentile / eigenSum
    if percentile/eigenSum > 0.90:
      component = i
      print "90 percent of the variance explained at component: " + str(i+1)
      break
  z = np.dot(meanRemoved,principleComponents)
  x =  z[:,0]
  y =  z[:,1]

  plot(x,y,shapes_signs,meanRemoved,principleComponents)

"""
Function for plotting the projected data and the 4 clusters
"""
def plot(x,y,label,meanRemoved,principleComponents):
  #print type(label)
  #print "second"
  label = (label.tolist())
  label = map(int, label)
  #print label
  #ylabel_ = label.tolist()
  #print type(ylabel_)
  for i in range(1):
    fig = plt.figure()
    plt.title("First two Principle Components")
    ax  = fig.add_subplot(111)
    #red_patch = mpatches.Patch(color='red', label='The red data')
    blue_line = mlines.Line2D([], [], color='red', marker='o',
                          markersize=5, label='Weed')
    green_line = mlines.Line2D([], [], color='green', marker='^',
                          markersize=5, label='Crop')
    c = ['red','green']
    m = ['o','^']
    al = [1,1,1,1,1,1]
    for j in range(len(x)):
      #rint type(label[j])
      #print j
      ax.scatter(x[j],y[j],c=c[label[j]],marker=m[label[j]],alpha=al[label[j]])
    #plt.legend(handles=[x],loc=3)
    clusters = km.kMeans(meanRemoved,2)
    reducedClusters = np.dot(clusters,principleComponents)
    x1 = reducedClusters[:,0]
    y1 = reducedClusters[:,1]
    cluster = ax.scatter(x1,y1,label='Clusters',c='blue',s=700,marker='.')
    #plt.legend(handles=[cluster],loc=3)
    #plt.savefig('kmeansAndTwpPCA')
    #plt.close()
    plt.legend(handles=[blue_line,green_line,cluster])
    #ax.legend(loc=0, scatterpoints = 1,label=1)
    #plt.legend()

    plt.show()

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
  #plt.savefig('eigenSpectrum')
  plt.show()


def main():
  print "Running PCA"
  getPCA("ML2016WeedCropTrain.csv1",13)

#main()