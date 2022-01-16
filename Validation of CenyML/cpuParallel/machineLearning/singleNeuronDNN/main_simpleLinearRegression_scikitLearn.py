# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: César Miranda Meza
# COMPLETITION DATE: December 27, 2021.
# LAST UPDATE: N/A
#
# This code is used to apply the machine learning method known as Multilayer
# Perceptron Regressor as made available by the scikit-learn library
# (https://bit.ly/3J9Vv0y), but with only two single neurons as this apparently
# seems to be the minimum number of neurons that can be defined in a model with
# such method. The resulting model will be used to solve a simple linear
# regression problem. This is done with the database used for linear equation
# systems. In addition, this database has 1'000'000 samples. Then, some metrics
# will be obtained, along with a plot to use these as a comparative evaluation
# of the results obtained in the CenyML library.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Python version 3.9.7


# ----------------------------------- #
# ----- Importing the Libraries ----- #
# ----------------------------------- #
import pandas as pd  # version 1.3.3
import numpy as np # version 1.21.2
import time
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error # version 1.0.1
import matplotlib.pyplot as plt # version 3.4.3


# -------------------------------------------- #
# ----- Define the user variables values ----- #
# -------------------------------------------- #
m = 1 # This variable is used to define the number of independent variables
      # that the system under study has.
p = 1 # This variable is used to define the number of dependent variables
      # that the system under study has.
idealCoefficients = [[10], [0.8]] # This variable will store the ideal
                                  # coefficient values that it is expected for
                                  # this model to have.
columnIndexOfOutputDataInCsvFile = 2; # This variable will contain the index
                                      # of the first column in which we will
                                      # specify the location of the output
                                      # values (Y and/or Y_hat).
columnIndexOfInputDataInCsvFile = 3; # This variable will contain the index
                                     # of the first column in which we will
                                     # specify the location of the input
                                     # values (X).
                                     
                                     
# ------------------------------ #
# ----- Import the dataset ----- #
# ------------------------------ #
# Read the .csv file containing the data to be trained with.
print("Innitializing data extraction from .csv file containing the data to train with ...")
startingTime = time.time()
dataset_rLES1000S1000SPS = pd.read_csv("../../../../databases/regression/linearEquationSystem/1000systems_1000samplesPerSys.csv")
elapsedTime = time.time() - startingTime
n = len(dataset_rLES1000S1000SPS)
csvColumns = len(dataset_rLES1000S1000SPS.iloc[0])
print("Data extraction from .csv file containing " + format(n) + " samples for each of the " + format(csvColumns) + " columns (total samples = " + format(n*csvColumns) + ") elapsed " + format(elapsedTime) + " seconds.")
print("")


# ------------------------------------- #
# ----- Preprocessing of the data ----- #
# ------------------------------------- #
# Retrieving the real data of its corresponding dataset
print("Innitializing input and output data with " + format(n) + " samples for each of the " + format(p) + " columns (total samples = " + format(n*p) + ") ...")
startingTime = time.time()
X = np.ones((n, 0))
Y = np.ones((n, 0))
for currentColumn in range(0, m):
    temporalRow = dataset_rLES1000S1000SPS.iloc[:,columnIndexOfInputDataInCsvFile].values.reshape(n, 1)
    X = np.append(X, temporalRow, axis=1)
    temporalRow = dataset_rLES1000S1000SPS.iloc[:,columnIndexOfOutputDataInCsvFile].values.reshape(n, 1)
    Y = np.append(Y, temporalRow, axis=1)
elapsedTime = time.time() - startingTime
print("Input and output data innitialization elapsed " + format(elapsedTime) + " seconds.")
print("")


# -------------------------- #
# ----- Model training ----- #
# -------------------------- #
print("Innitializing model training with the scikit-learn library ...")
startingTime = time.time()
# NOTE: Hidden layers must be > 0 in scikit-learn and, therefore, no single neuron model can be generated with this library.
# NOTE: hidden_layer_sizes = ("desired neurons in first hidden layer", "desired neurons in second hidden layer", ..., "desired neurons in last hidden layer")
MLP = MLPRegressor(hidden_layer_sizes=(1), activation='identity', solver='sgd', learning_rate_init=0.000000000001, max_iter=30863, shuffle=False, random_state=0)
multiLayerPerceptron = MLP.fit(X, Y.ravel())
elapsedTime = time.time() - startingTime
print("Model training with the scikit-learn library elapsed " + format(elapsedTime) + " seconds.")
print("")


# ------------------------------------------- #
# ----- Storage of the results obtained ----- #
# ------------------------------------------- #
# We obtained the predicted results by the model that was constructed.
print("Innitializing predictions of the model obtained ...")
startingTime = time.time()
Y_hat = multiLayerPerceptron.predict(X)
elapsedTime = time.time() - startingTime
print("Model predictions elapsed " + format(elapsedTime) + " seconds.")
print("")

# We obtained the mean squared error metric on the obtained ML model.
print("Innitializing scikit-learn mean squared error metric calculation ...")
startingTime = time.time()
MSE = mean_squared_error(Y, Y_hat) # We apply the desired evaluation metric.
elapsedTime = time.time() - startingTime
print("MSE = " + format(MSE))
print("scikit-learn mean squared error metric elapsed " + format(elapsedTime) + " seconds.")
print("")

# We visualize the trainning set results
X_plot = dataset_rLES1000S1000SPS.independent_variable_1
plt.scatter(X_plot, Y, color='red')
plt.plot(X_plot, Y_hat, color='blue')
plt.title('Simple linear regression with the \"Sequential()\" neural network of tensorflow')
plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')

# We display, in console, the coefficient values obtained with the ML method used.
# NOTE: The bias and weight values will be stored in "b" such that each column
#       will contain the bias and weight values of a certain neuron.
b = np.zeros((2, m+1))
# NOTE: For both "multiLayerPerceptron.intercepts_" and
#       "multiLayerPerceptron.coefs_", it seems that the first array will
#       always refer to the first defined hidden layer and the last array to
#       the output layer. In addition and apparently, the input layer either
#       does not have neurons defined in it or the first defined hidden layer
#       is considered as the input layer by scikit-learn.
# NOTE: "multiLayerPerceptron.intercepts_" will display several arrays, each
#       for a different layer. Within each array/layer, each column will stand
#       for the bias value of a different neuron.
b[0][0] = multiLayerPerceptron.intercepts_[0][0]
b[0][1] = multiLayerPerceptron.intercepts_[1][0]
# NOTE: "multiLayerPerceptron.coefs_" will display several arrays, each for a
#       different layer. Within each array/layer, each column will stand for
#       the weight values of a different neuron.
b[1][0] = multiLayerPerceptron.coefs_[0][0][0]
b[1][1] = multiLayerPerceptron.coefs_[1][0][0]
print("b_0 = " + format(b[0][0]))
print("b_1 = " + format(b[0][1]))

print("The program has finished successfully.")
