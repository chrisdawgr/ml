import matplotlib.pyplot as plt

from numpy import loadtxt, array, sqrt, sin, average, zeros
from numpy import subtract, dot, add, mean, var, arange
from numpy.random import rand, seed, sample

from copy import copy

TrainData = loadtxt("sincTrain25.dt", usecols=(0, ))
TrainTarget = loadtxt("sincTrain25.dt", usecols=(1, ))
ValidateData = loadtxt("sincValidate10.dt", usecols=(0, ))
ValidateTarget = loadtxt("sincValidate10.dt", usecols=(1, ))

seed(10)
trainInterval = arange(-10, 10, 0.05, dtype='float64')

"""
Calculates the linear model target value based on the given weights and bias.
Used for hidden units, when firing towards the output unit
"""
def calc_output(hidden_data, weight, bias):
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
def feedForward(data, weights_m, weights_k, num_hidden, num_input=1,\
                num_output=1):
  prediction = []
  for x in data:
    # for each data entry
    hidden_data = []
    for h_unit in xrange(num_hidden):
      # call activation function, with weights for hidden neurons
      hidden_data.append(calc_hidden(x, weights_m[h_unit],1))
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

def backPropagation(train, target, steps, learning_rate, weights_m,\
                    weights_k, num_hidden):
  valData    = ValidateData
  valTarget  = ValidateTarget
  errors     = []; val_errors = []
  for _ in xrange(steps):
    # perform forward propagation
    # for validation only perform foward with given weights
    predicted = feedForward(train, weights_m, weights_k, num_hidden)
    predicted_interval = feedForward(trainInterval, weights_m,\
      weights_k, num_hidden)
    predicted_validation = feedForward(valData, weights_m, weights_k,\
                                       num_hidden)
    d_hidden = []
    d_out = []
    for i, data in enumerate(train):
      dK = delta_k(predicted[i], target[i])
      dJs = []
      list_zj = []
      for j in xrange(num_hidden):
        aj = weights_m[j][0] * data + weights_m[j][1] * 1
        # calculate z
        list_zj.append(activation_function(aj))
        # get output neuron
        dJs.append(delta_j(aj, weights_k[0][j], [dK]))
      # append each hidden units output fire
      d_hidden.append(dot(array(dJs), array([[data, 1]])))
      # add activation for bias
      list_zj.append(array([activation_function(1)]))
      d_out.append(dK * array(list_zj))
    avg_dhidden = average(d_hidden, axis=0)
    avg_dout = average(d_out, axis=0).flatten()
    # update weights using grading decent
    weights_m = subtract(weights_m, (learning_rate * avg_dhidden))
    weights_k = subtract(weights_k, (learning_rate * avg_dout))
    # append errors for plotting
    #print "train: predicted: " + str(predicted) + " actual: " + str(target)
    #print (MSE(predicted, target))
    #print "2"
    #print MSE(predicted_validation,valTarget)
    errors.append(MSE(predicted, target))
    val_errors.append(MSE(predicted_validation,valTarget))
  return errors,val_errors, predicted_interval, avg_dhidden, avg_dout

def gradient_verify(data, data_target, weights_k, weights_m, e,num_hidden):
  error_matrix_md = zeros(weights_m.shape)
  error_matrix_km = zeros(weights_k.shape)
  errors, _, _, _, _ = backPropagation(data, data_target, 1, -1,\
                                    weights_m, weights_k,num_hidden)
  for i in xrange(weights_m.shape[1]):
    for j in xrange(len(weights_m)):
      cpy_weight_md = copy(weights_m)
      cpy_weight_md[j][i] += e
      e_errors, _,  _, _, _ = backPropagation(data, data_target, 1, -1,\
                                          cpy_weight_md, weights_k,num_hidden)
      error_matrix_md[j][i] = (e_errors[0][0] - errors[0][0]) / e
  for i in xrange(weights_k.shape[1]):
    cpy_weight_km = copy(weights_k)
    cpy_weight_km[0][i] += e
    e_errors, _, _, _, _ = backPropagation(data, data_target, 1, -1,\
                                        weights_m, cpy_weight_km,num_hidden)
    error_matrix_km[0][i] = (e_errors[0][0] - errors[0][0]) / e
  return error_matrix_md, error_matrix_km

#for num_hidden, learning_rates in [(2, [0.0001, 0.001, 0.01, 0.1, 1]), (20, [0.0001, 0.001, 0.01, 0.1])]:
def main(iterations):
  test_sets1        = [(2, [0.0001,0.01,0.1]),\
                      (20, [0.0001,0.01,0.1])]
  test_sets        = [(2, [0.0001, 1]),\
                      (20, [0.0001,0.1])]
  for num_hidden, learning_rates in test_sets1:
    # weights to hidden layers
    weights_m   = sample([num_hidden, 2])
    # weights from hidden to output
    weights_k   = sample([1, num_hidden + 1])

    error_matrix_md, error_matrix_km = gradient_verify(TrainData[:10],\
      TrainTarget[:10], weights_k, weights_m, 0.00000001,num_hidden)
    _, _, _, avg_dhidden, avg_dout =backPropagation(TrainData[:10],\
      TrainTarget[:10], 1, -1, weights_m, weights_k,num_hidden)

    errors_list = []
    errors_list_validate = []
    data_list = []
    for learning_rate in learning_rates:
      errors, val_errors, data, _, _ = backPropagation(TrainData, TrainTarget, \
                             iterations, learning_rate, weights_m,\
                             weights_k,num_hidden)
      errors_list.append(errors)
      errors_list_validate.append(val_errors)
      data_list.append(data)

    plt.gca().set_yscale('log')
    #TODO plot 2 and 20 in same plot
    #plt.set_yscale('log')

    for index, err in enumerate(errors_list):
      plt.plot(range(len(err)), err, label="Train: " +\
               str(learning_rates[index]))
    for index, err in enumerate(errors_list_validate):
      plt.plot(range(len(err)), err, label="Val: " +\
               str(learning_rates[index]))
    plt.rc('text', usetex=True)
    plt.rc('font', family='Computer Modern',size=12)
    plt.xlabel(r'\textit{Iterations} ($\epsilon$)')
    plt.ylabel(r'\textit{MSE')
    plt.legend(loc=4)
    plt.show()

    for index, da in enumerate(data_list):
      plt.plot(trainInterval, da, label="Learning rate " +\
               str(learning_rates[index]))
    plt.plot(trainInterval, eval('sin(trainInterval)/trainInterval'\
             .format(trainInterval)), label="sin(x)/x")
    plt.scatter(TrainData, TrainTarget)
    plt.legend()
    plt.show()
if __name__ == '__main__':
  main(1000)