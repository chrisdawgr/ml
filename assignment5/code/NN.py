import matplotlib.pyplot as plt
import sys
from numpy import loadtxt, array, sqrt, sin, average, zeros
from numpy import subtract, dot, add, mean, var, arange, delete
from numpy.random import rand, seed, sample

from copy import copy

"""
Retrieve data and target values for the training and validation set
"""
def prepareData(trainfile, validatefile):
  training_set = loadtxt(trainfile)
  validation_set = loadtxt(validatefile)
  num_features = len(training_set[0])-1
  TrainData = delete(training_set, num_features, axis=1)
  TrainTarget = training_set[:,num_features]
  ValidateData = delete(validation_set, num_features, axis=1)
  ValidateTarget = validation_set[:,num_features]
  return TrainData,TrainTarget,ValidateData,ValidateTarget

TrainData,TrainTarget,ValidateData,ValidateTarget = \
  prepareData("sincTrain25.dt","sincValidate10.dt")

seed(10)
trainInterval = arange(-10, 10, 0.05, dtype='float64')

"""
Calculates the linear model target value based on the given weights and bias.
Used for hidden units, when firing towards the output unit
"""
def calc_output(hidden_data, weight, bias):
  #print hidden_data + [bias]
  return dot(weight, hidden_data + [bias])

"""
Caluclates the linear model of the data, and applies the activation function
/transfer function
"""
def calc_hidden(data, weight, bias):
  return activation_function(weight[0] * data + weight[1] * bias)

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
Compute feed-forward network. For each entry point in the data, the hidden
values of a is calculated based on the each input, the output for each entry
is then calculated using the activation function
"""
def feedForward(data, weights_m, weights_k, num_input=1,\
                num_output=1):
  prediction = []
  for x in data:
    # for each data entry
    hidden_data = []
    for h_unit in xrange(num_hidden):
      # call activation function, with weights for hidden neurons
      hidden_data.append(calc_hidden(x, weights_m[h_unit],1))
      #print weights_m
      print weights_k
    # get output using the weights from the output neurons
    prediction.append(calc_output(hidden_data, weights_k, 1))
  return prediction

"""
Compute the mean squared error on the predicted values and the targetvalues
"""
def MSE(data, t):
  return 1 / float(2 * len(data)) * sum((t[i] - data[i]) ** 2\
    for i, d in enumerate(data))


"""
Inner derivative using sum rule
"""
def delta_k(predicted, target):
  return predicted - target

"""
Outer derivative using chain rule
"""

def delta_j(a, weight, delta_k):
  return act_function_deriv(a) * sum([weight * data_k for k, data_k\
    in enumerate(delta_k)])

def backPropagation(learning_rate,weights_m,weights_k,verify=False,steps=4000):
  valData    = ValidateData
  valTarget  = ValidateTarget
  train      = TrainData
  target     = TrainTarget
  if verify:
    train  = train[:10]
    target = target[:10]
  errors     = []; val_errors = []
  p = 1
  steps = 1
  for _ in xrange(steps):

    if p % steps == 0:
      sys.stdout.write('\r'+"")
      sys.stdout.flush()
    else:
      if p % (steps/100.0) == 0:
        b =  (str(int(float(p) / steps * 100)) + " %")
        sys.stdout.write('\r'+str(b))
        sys.stdout.flush()
    predicted = feedForward(train, weights_m, weights_k)
    predicted_interval = feedForward(trainInterval, weights_m,\
      weights_k)
    predicted_validation = feedForward(valData, weights_m, weights_k)
    d_hidden = []
    d_out = []
    for i, data in enumerate(train):
      dK = delta_k(predicted[i], target[i])
      dJs = []
      list_zj = []
      for j in xrange(num_hidden):
        aj = weights_m[j][0] * data + weights_m[j][1] * 1
        list_zj.append(activation_function(aj))
        dJs.append(delta_j(aj, weights_k[0][j], [dK]))
      d_hidden.append(dot(array(dJs), array([[data, 1]])))
      list_zj.append(array([activation_function(1)]))
      d_out.append(dK * array(list_zj))
    avg_dhidden = average(d_hidden, axis=0)
    avg_dout = average(d_out, axis=0).flatten()
    weights_m = subtract(weights_m, (learning_rate * avg_dhidden))
    weights_k = subtract(weights_k, (learning_rate * avg_dout))
    errors.append(MSE(predicted, target))
    val_errors.append(MSE(predicted_validation,valTarget))
    p += 1
  return errors,val_errors, predicted_interval, avg_dhidden, avg_dout

def gradient_verify(weights_k, weights_m, e=0.00000001):
  data  = TrainData[:10]
  data_target = TrainTarget[:10]

  error_matrix_md = zeros(weights_m.shape)
  error_matrix_km = zeros(weights_k.shape)
  errors, _, _, _, _ = backPropagation(-1,\
                                    weights_m, weights_k,True,1)
  for i in xrange(weights_m.shape[1]):
    for j in xrange(len(weights_m)):
      cpy_weight_md = copy(weights_m)
      cpy_weight_md[j][i] += e
      e_errors, _,  _, _, _ = backPropagation(-1,\
                                          cpy_weight_md, weights_k,True,1)
      error_matrix_md[j][i] = (e_errors[0][0] - errors[0][0]) / e
  for i in xrange(weights_k.shape[1]):
    cpy_weight_km = copy(weights_k)
    cpy_weight_km[0][i] += e
    e_errors, _, _, _, _ = backPropagation(-1,\
                                        weights_m, cpy_weight_km,True,1)
    error_matrix_km[0][i] = (e_errors[0][0] - errors[0][0]) / e
  return error_matrix_md, error_matrix_km

def main(iterations):
  test    = [(2, [0.1])]
  twoPlot = []
  global num_hidden
  for num_hidden, learning_rates in test:
    print "Number of hidden: " +  str(num_hidden)
    # weights to hidden layers
    weights_m   = sample([num_hidden, 2])
    # weights from hidden to output
    weights_k   = sample([1, num_hidden + 1])
    #error_matrix_md, error_matrix_km = gradient_verify(weights_k, weights_m)

    #_, _, _, avg_dhidden, avg_dout = backPropagation(-1, weights_m,\
    #                                                 weights_k,True,1)
    # gradient verification, values should be within 10^-8
    #print subtract(error_matrix_md, avg_dhidden)
    #print subtract(error_matrix_km, avg_dout)

    errors_list = [];errors_list_validate = []; data_list = []
    for learning_rate in learning_rates:
      print "Learning Rate: " +str(learning_rate)
      errors, val_errors, data, _, _ = \
        backPropagation(learning_rate, weights_m, weights_k,False,iterations)
      errors_list.append(errors)
      errors_list_validate.append(val_errors)
      data_list.append(data)

      if learning_rate == 0.1:
        twoPlot.append(errors)

    """
    plt.gca().set_yscale('log')
    # Plot of all learning rates and validation errors
    for index, err in enumerate(errors_list):
      plt.plot(range(len(err)), err, label="Train: " +\
                str(learning_rates[index]))
      plt.plot(range(len(err)), errors_list_validate[index] , label="Val: " +\
               str(learning_rates[index]))
    plt.rc('text', usetex=True)
    plt.rc('font', family='Computer Modern',size=12)
    plt.xlabel(r'\textit{Iterations} ($\epsilon$)')
    plt.ylabel(r'\textit{MSE')
    header = "Hidden units: " + str(num_hidden)
    plt.title(header)
    plt.legend(loc=1,prop={'size':10})
    title = "img/rates-"+str(num_hidden)+str("-")+str(learning_rates)+str("-")+str(iterations)+str(".png")
    plt.savefig(title)
    plt.close()

    # Plot of interval
    for index, da in enumerate(data_list):
      plt.plot(trainInterval, da, label="Learning rate: " +\
               str(learning_rates[index]))
    plt.plot(trainInterval, eval('sin(trainInterval)/trainInterval'\
             .format(trainInterval)), label="sin(x)/x")
    plt.rc('text', usetex=True)
    plt.rc('font', family='Computer Modern',size=12)
    #plt.xlabel(r'\textit{Iterations} ($\epsilon$)')
    #plt.ylabel(r'\textit{MSE')
    plt.scatter(TrainData, TrainTarget, label="Training Data")
    plt.scatter(ValidateData,ValidateTarget, label="Validation Data",color='red')
    plt.legend(loc=1,prop={'size':10})
    header = "Hidden units: " + str(num_hidden)
    plt.title(header)
    title = "img/interval"+str(num_hidden)+str("-")+str(learning_rates)+str(".png")
    plt.savefig(title)
    plt.close()
  # Plot of 2 and 20
    plt.gca().set_yscale('log')
  for index, err in enumerate(twoPlot):
    if index == 0:
      labelhere = "2 hidden: "
    else:
      labelhere = "20 hidden: "
    plt.plot(range(len(err)), err, label=labelhere + str(0.1))
  plt.xlabel(r'\textit{Iterations} ($\epsilon$)')
  plt.ylabel(r'\textit{MSE')
  plt.legend(loc=1)
  title = "img/twoplots"+str(num_hidden)+str("-")+str(learning_rates)+str(".png")
  plt.savefig(title)
  plt.close()
  """
main(1)