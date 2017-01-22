from svmutil import *
import numpy as np
import math

def prepareData_(Data):
  numFeatures = Data.shape[1]-1
  targetvalues = Data[:,numFeatures]
  features = np.delete(Data, numFeatures, axis=1)
  return(targetvalues,features)

def Norm(train,test):
  Data = np.loadtxt(train)
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
  return normalizedTest,testval,normalizedData,val

# for each pair gamma, c, perform 5 cross valdation
def svm(dat_x,dat_y,val_x,val_y,norm,c,g):
  # train according to the data
  prob = svm_problem(dat_y,dat_x)
  param = svm_parameter('-s 0 -t 2 -c %f -g %f -q' % (c,g))
  m = svm_train(prob, param)
  # predict on validation set
  p_labels, p_acc, p_vals = svm_predict(val_y, val_x, m)
  return p_acc[0]

def removeRest(i,listofarrays):
  trainingset = []
  validation = listofarrays[i]
  for j in range(len(listofarrays)):
    if i != j:
      trainingset.append(listofarrays[j])
  assert len(listofarrays) == 5, "Not implemented for sizes != 5"
  comb= np.vstack([trainingset[0],trainingset[1],trainingset[2],\
                   trainingset[3]])
  return (validation,comb)

def generaretSplits(Data,splits):
  folds = np.array_split(Data,splits)
  combList = []
  for i in range(len(folds)):
    combList.append(removeRest(i,folds))
  return combList

def single_pass(x,y,x_val,y_val):
  jaakalov = findG()
  #b = 10
  b = 2
  #b = math.exp(1)
  C = [b**-3,b**-2,b**-1,1,b,b**2,b**3,b**4,b**5,b**6]
  gamma1 = [-3,-2,-1,0,1,2,3]
  gamma = [jaakalov * b**n for n in gamma1]
  # keep track of best result to determine the best
  zeroList = []
  best_res = 0
  best_c = 0
  best_g = 0
  mat = np.zeros([len(C),len(gamma)])
  for n, c in enumerate(C):
    for m,g in enumerate(gamma):
      result = svm(x,y,x_val,y_val,False,c,g)
      mat[n,m] = result
      if result > best_res:
        best_res = result
        best_g   = g
        best_c   = c
  return mat,best_res,(best_c,best_g)

"""
perform 5-fold cross validation and report each matlist, best result and bestCg
C, gamma
"""
def crossValidate(splits,norm):
  matlist = []
  Train = np.loadtxt("ML2016WeedCropTrain.csv1")
  (y,x) = prepareData_(Train)
  if norm:
    (_,_,x,y) = Norm("ML2016WeedCropTrain.csv1","ML2016WeedCropTest.csv1")
  folds = generaretSplits(x,splits)
  y = y.reshape(len(y),1)
  yfolds = generaretSplits(y,splits)
  bestRes = 0
  bestCg = 0
  # for each partition
  for i in range(len(folds)):
    val = folds[i][0]
    print "HER"
    train = folds[i][1]
    target_val = yfolds[i][0]
    target_train = yfolds[i][1]

    x_val = val.tolist()
    x_train = train.tolist()
    y_val = target_val.tolist()
    y_val = list_(y_val)
    y_train = target_train.tolist()
    y_train = list_(y_train)
    mat,result, cg = single_pass(x_train,y_train,x_val,y_val)
    matlist.append(np.array(mat))
    print "RES: " + str(bestRes)
    if result > bestRes:
      bestRes = result
      bestCg = cg
  return matlist,bestRes,bestCg

def getAvg(listr):
  C_min, C_max = -2,8
  gamma_min, gamma_max = -6,4
  C = [10**n for n in range(C_min,C_max)]
  gamma = [10**n for n in range(gamma_min,gamma_max)]
  n,m = listr[0].shape
  summed_arr = np.zeros([n,m])
  for arr in listr:
    summed_arr = np.add(summed_arr,arr)
  summed_arr = summed_arr / 5.0
  biggest = 0
  index = 0
  for i in range(n):
    for j in range(m):
      if summed_arr[i][j] > biggest:
        biggest = summed_arr[i][j]
        index = (i,j)
  q,w = index
  return index,biggest

def list_(inp):
  retList = []
  for i in range(len(inp)):
    retList.append(inp[i][0])
  return retList

def test(norm):
  mat,b, _ = crossValidate(5,norm)
  (c,g), error = getAvg(mat)
  print "INDEX, c: " +str(c)

  print "INDEX, g: " +str(g)
  jaakalov = findG()
  b = 2
  #b = 10
  #b = math.exp(1)
  C = [b**-3,b**-2,b**-1,1,b,b**2,b**3,b**4,b**5,b**6]
  gamma1 = [-3,-2,-1,0,1,2,3]
  gamma = [jaakalov * b**n for n in gamma1]
  c = C[c]
  g = gamma[g]
  print "C: " + str(c)
  print "g: " + str(g)
  #c = 1
  #g = 0.1
  Data = np.loadtxt("ML2016WeedCropTrain.csv1")
  DataVal = np.loadtxt("ML2016WeedCropTest.csv1")
  (y,x) = prepareData_(Data)
  (y_val,x_val) = prepareData_(DataVal)
  if norm:
    (x_val,y_val,x,y) = Norm("ML2016WeedCropTrain.csv1",\
                            "ML2016WeedCropTest.csv1")
  (y_,x_) = (y,x)
  x = x.tolist()
  y =  y.tolist()
  y_val = y_val.tolist()
  x_val = x_val.tolist()
  result = (svm(x,y,x_val,y_val,False,c,g))
  result1 = (svm(x,y,x,y,False,c,g))
  #result1 = 1 -  (float(result1)/100)
  print result1

  return 1 - (float(result1)/100), 1- (float(result)/100),(c,g)

def getDistanceTwoPoints(x,xi):
  inner = 0
  for i in range(len(x)):
    #print x[i] - xi[i]
    inner += (x[i] - xi[i])**2
  return math.sqrt(inner)

# Calculate the distance vector for a point and a vector of points
def getDistanceMat(x,Mat):
  returnArray = np.zeros(len(Mat))
  for i in range(len(Mat)):
    returnArray[i] = getDistanceTwoPoints(x,Mat[i])
  return returnArray


def findG():
  G = 0
  G1 = []
  Train = np.loadtxt("ML2016WeedCropTrain.csv1")
  (targets,feats) = prepareData_(Train)
  print len(feats)
  for i in range(len(feats)):
    target = targets[i]
    distanceArray = getDistanceMat(feats[i],feats)
    sortedIndecies = distanceArray.argsort()
    #print sortedIndecies
    for j in range(len(distanceArray)):
      if targets[sortedIndecies[j]] != target:
        #print getDistanceTwoPoints(feats[i],
        #      feats[sortedIndecies[j]])

        G += getDistanceTwoPoints(feats[i],
              feats[sortedIndecies[j]])
        G1.append(getDistanceTwoPoints(feats[i],
              feats[sortedIndecies[j]]))
        break
  G1 = np.array(G1)
  print len(G1)
  print "MEAN"
  print np.mean(G1)
  sig_jaakola = float(G) / len(feats)
  print "sig"
  print sig_jaakola
  gamma_jaakola = 1/(2*sig_jaakola**2)
  return gamma_jaakola

# G = 3372.82250236

def main():
  trainError,valError,(c,g) = test(True)
  #b, (c1,g1) = test(True)
  print "hyperparameter c: " + str(c)
  print "hyperparameter g: " + str(g)
  print "Training error: " + str(trainError)
  print "Test error: " + str(valError)
  #print "Generalization error, norm data: " + str(b)

main()