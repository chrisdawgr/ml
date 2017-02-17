from svmutil import *
import numpy as np

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
  prob = svm_problem(dat_y,dat_x)
  param = svm_parameter('-s 0 -t 2 -c %f -g %f -q' % (c,g))
  m = svm_train(prob, param)
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
  C_min, C_max = -2,8
  gamma_min, gamma_max = -6,4
  C = [10**n for n in range(C_min,C_max)]
  gamma = [10**n for n in range(gamma_min,gamma_max)]
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

def crossValidate(splits,norm):
  matlist = []
  Train = np.loadtxt("parkinsonsTrainStatML.dt")
  (y,x) = prepareData_(Train)
  if norm:
    (_,_,x,y) = Norm("parkinsonsTrainStatML.dt","parkinsonsTestStatML.dt")
  folds = generaretSplits(x,splits)
  y = y.reshape(len(y),1)
  yfolds = generaretSplits(y,splits)
  bestRes = 0
  bestCg = 0
  for i in range(len(folds)):
    val = folds[i][0]
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
  c = 1
  g = 0.1
  Data = np.loadtxt("parkinsonsTrainStatML.dt")
  DataVal = np.loadtxt("parkinsonsTestStatML.dt")
  (y,x) = prepareData_(Data)
  (y_val,x_val) = prepareData_(DataVal)
  if norm:
    (x_val,y_val,x,y) = Norm("parkinsonsTrainStatML.dt",\
                            "parkinsonsTestStatML.dt")
  (y_,x_) = (y,x)
  x = x.tolist()
  y =  y.tolist()
  y_val = y_val.tolist()
  x_val = x_val.tolist()
  result = (svm(x,y,x_val,y_val,False,c,g))
  return 1- (float(result)/100),(c,g)

def main():
  a,(c,g) = test(False)
  b, (c1,g1) = test(True)
  print "hyperparameter c: " + str(c)
  print "hyperparameter g: " + str(g)
  print "Generalization error, raw data: " + str(a)
  print "Generalization error, norm data: " + str(b)