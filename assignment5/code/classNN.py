import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt

"""
The activation/transfer function
"""
def activation_function(a):
  return a / float(1 + abs(a))

"""
The derivative of the activation/transfer function

"""
def act_function_deriv(a):
  return 1 / float((1 + abs(a))) ** 2

"""
Compute the mean squared error on the predicted values and the targetvalues
"""
def MSE(data, t):
  return 1 / float(2 * len(data)) * sum((t[i] - data[i]) ** 2\
    for i, d in enumerate(data))

#TODO add bias
class NeuralNetwork(object):
  def __init__(self, numInput,numOutput,steps):
    self.numInput = numInput
    self.numHidden = None
    self.numOutput = numOutput
    self.training_X = None
    self.training_Y = None
    self.validation_X = None
    self.validation_Y = None
    self.weights_1 = None
    self.weights_2 = None
    self.steps     = steps
    self.eps       = None
    self.act = np.vectorize(activation_function)
    self.actprime = np.vectorize(act_function_deriv)
    self.startingWeights_1 = None
    self.startingWeights_2 = None

    self.InitializeData("sincTrain25.dt","sincValidate10.dt")

  def InitializeData(self,trainfile, validatefile):

    training_set = np.loadtxt(trainfile)
    validation_set = np.loadtxt(validatefile)
    num_features = len(training_set[0])-1

    self.training_X = np.delete(training_set, num_features, axis=1)
    training_Y = training_set[:,num_features]
    validation_Y = validation_set[:,num_features]
    self.validation_X = np.delete(validation_set, num_features, axis=1)
    self.validation_Y = validation_Y.reshape(len(validation_set),1)
    self.training_Y = training_Y.reshape(len(training_set),1)

  def initializeWeights(self):
    print "initializing weights"
    self.weights_1 = np.random.randn(self.numInput+1,self.numHidden)
    #self.weights_2 = np.random.randn(self.numHidden,self.numOutput+1)
    self.weights_2 = np.random.randn(self.numHidden+2,self.numOutput)
    self.startingWeights_1 = self.weights_1
    self.startingWeights_2 = self.weights_2

  def forwardPropagation(self,train):
    if (train):
      X = self.training_X
    else:
      X = self.validation_X
    X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
    Z_2 = np.dot(X,self.weights_1)
    a_2 = self.act(Z_2)
    a_2_1 = np.hstack((a_2, np.ones((a_2.shape[0], 2), dtype=a_2.dtype)))
    #print a_2
    #print a_2.shape
    #print self.weights_2.shape
    Z_3 = np.dot(a_2_1,self.weights_2)
    yhat = self.act(Z_3)
    return yhat, Z_2, a_2_1, Z_3

  def backwardPropagation(self):
    y_hat, Z_2, a_2, Z_3 = self.forwardPropagation(True)
    y_hat_val, _, _, _ = self.forwardPropagation(False)
    delta_3 = np.multiply(-(self.training_Y-y_hat), self.actprime(Z_3))
    weights_2 = self.weights_2[:2]
    dj_dw_2 = np.dot(a_2.T, delta_3)

    delta_2 = np.dot(delta_3,weights_2.T) * self.actprime(Z_2)
    dj_dw_1 = np.dot(self.training_X.T,delta_2)

    self.weights_1 = self.weights_1 - (np.dot(self.eps,dj_dw_1))
    self.weights_2 = self.weights_2 - (np.dot(self.eps,dj_dw_2))
    return MSE(y_hat,self.training_Y),MSE(y_hat_val,self.validation_Y)

  def train(self):
    errorList = []
    valList = []
    for _ in range(self.steps):
      train_error, val_error = self.backwardPropagation()
      errorList.append(train_error)
      valList.append(val_error)
    return errorList,valList

  def main(self):
    test = [(2, [0.01])]
    for numHidden, eps_ in test:
      self.numHidden = numHidden
      self.initializeWeights()
      errorList = []
      valList_ = []
      for eps in eps_:
        self.eps = eps
        errors_list, valList = self.train()
        errorList.append(errors_list)
        valList_.append(valList)
      e = np.array(errorList).reshape(1,self.steps,1)
      v = np.array(valList_).reshape(1,self.steps,1)
      print "End train error: " + str(e[0][-1])
      print "End valid error: " + str(v[0][-1])
      #if (v[0][-1]) < 0.020:
      #print e.shape
      #print len(errorList)

      #print v.shape
      self.plot(e,v,eps_)

  def plot(self,errors_list,valList,eps_):
    for index, err in enumerate(errors_list):
      plt.plot(range(len(err)), err, label="Train: " + str(eps_[index]))
      plt.plot(range(len(err)), valList[index] , label="Val: " +\
             str(eps_[index]))
    plt.gca().set_yscale('log')
    plt.rc('text', usetex=True)
    plt.rc('font', family='Computer Modern',size=12)
    plt.xlabel(r'\textit{Iterations} ($\epsilon$)')
    plt.ylabel(r'\textit{MSE')
    header = "Hidden units: " + str(self.numHidden)
    plt.title(header)
    plt.legend(loc=1,prop={'size':10})
    title = "img/rates-"+str(self.numHidden)+str("-")+str(self.eps)+str("-")+str(self.steps)+str(".png")
    plt.plot()
    plt.show()

for i in range(0,10):
  NN = NeuralNetwork(1,1,3200)
  NN.main()