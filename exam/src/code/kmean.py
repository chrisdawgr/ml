import numpy as np
import math

"""
Get Euclidean distance
"""
def dist_euclid(vec1, vec2):
  return np.sqrt(np.sum(np.power(vec1-vec2,2)))

"""
Initialize the k clusters to have the first k datapoints
"""
def initialize_centroids(features,k):
  centroidList = []
  for i in range(k):
    centroidList.append(features[i])
  return centroidList

"""
Assign the clusters random starting points of the datapoints
"""
def initialize_centroids_random(features,k):
  centroidList = []
  indexlist= []
  for i in range(k):
    index = np.random.randint(0, len(features))
    centroidList.append(features[index])
    indexlist.append(index)
  return centroidList

"""
Perform k-means clustering for the input data and k clusters, returns the
position of these clusters
"""
def kMeans(data,k):
  features = data
  centroidList = np.array(initialize_centroids(features,k))
  moving = True
  j = 1
  while moving:
    belongsToCentroid = []
    # create a list for each centroid, containing points closest to
    for i in range(len(centroidList)):
      belongsToCentroid.append([])
    # use dictionary instead
    for p in range(len(features)):
      lowestDist = float("inf")
      picked_centroid = -1
      # for each point in the dataset, calculate distance to each centroid
      # add to centroid list with lowest distance
      for c in range(len(centroidList)):
        distance_to_centroid = dist_euclid(features[p],centroidList[c])
        if distance_to_centroid < lowestDist:
          # update lowest distance
          lowestDist = distance_to_centroid
          picked_centroid = c
      # add point to centroid list
      belongsToCentroid[picked_centroid].append(features[p])
    # array holds for each centroid the points closest to it
    newarray = np.array(belongsToCentroid)
    meandif = []
    for centr in range(len(newarray)):
      centrmean = np.mean(newarray[centr],axis=0)
      prevMean = np.sum(centroidList[centr])
      centroidList[centr] = centrmean
      meandif.append(abs(prevMean - np.sum(centroidList[centr])))
    if (all(x == 0.0 for x in meandif)):
      moving = False
      print "stopped after iteration: " +str(j)
    j += 1
  return centroidList