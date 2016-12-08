import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go

import PCA as pca

# initilize k clusters randomly
# calculate distance for each point to every cluster
# assign neares cluster
# calculate mean of points within cluster
# move cluster to said mean, recaluclate

def initialize_centroids(features,k):
  centroidList = []
  for i in range(k):
    centroidList.append(features[i])

def kMeans(filename,k):
  features, targets = pca.prepareData(filename)
  centroidList = initialize_centroids(features,k)

  


