"""
PACKAGE INSTALLATION STEPS:
    1) Open Anaconda Prompt terminal window.
    2) Type the command "conda create --name py3_9_7 python=3.9.7"
    3) Type the command "conda activate py3_9_7"
    4) Type the command "conda install pip"
    5) Type the command "pip install tensorflow==2.6"
    6) Type the command "pip install scikit-learn==1.0.1"
    7) Type the command "pip install numpy==1.21.2"
    8) Type the command "pip install matplotlib==3.4.3"
    9) Type the command "pip install pandas==1.3.3"
    10) Type the command "pip install spyder" (version installed = 5.2.0)
    11) Type the command "spyder"
    12) run the program.
"""
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: César Miranda Meza
# COMPLETITION DATE: November 29, 2021.
# LAST UPDATE: N/A
#
# This code is used to apply the machine learning method known as artificial
# neural network but with only a single neuron to solve a simple linear
# regression problem. This is done with the database used for linear equation
# systems. In addition, this database has 1'000'000 samples. Moreover, the
# well known tensorflow library will be used for such machine learning
# algorithm (https://keras.io/api/). Then, some metrics will be obtained,
# along with a plot to use these as a comparative evaluation of the results
# obtained in the CenyML library.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Python version 3.9.7


# ----------------------------------- #
# ----- Importing the Libraries ----- #
# ----------------------------------- #
import pandas as pd  # version 1.3.3
import numpy as np # version 1.21.2
import time
from numpy import loadtxt # version 1.21.2
from keras.models import Sequential # version 2.6.0
from keras.initializers import Zeros # version 2.6.0
from keras.layers import Dense # version 2.6.0
from keras.callbacks import EarlyStopping # version 2.6.0
from tensorflow.keras.optimizers import SGD # version 2.6.0
from sklearn.metrics import mean_squared_error # version 1.0.1
import matplotlib.pyplot as plt # version 3.4.3
import tensorflow as tf # version 2.6.0


# -------------------------------------------- #
# ----- Define the user variables values ----- #
# -------------------------------------------- #
m = 1 # This variable is used to define the number of independent variables
      # that the system under study has.
p = 1 # This variable is used to define the number of dependent variables
      # that the system under study has.
nodes = 1 # This variable will indicate the number of nodes that are desired
          # for the artificial neural network to be created.
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
print("Innitializing model training with the tensorflow library ...")
startingTime = time.time()
# Define the number of threads that are desired to be used (the value of 0
# makes tensorflow automatically define the most suitable number).
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
with tf.device('/CPU:0'): # To define the specific CPU to be used.
#tf.config.threading.set_intra_op_parallelism_threads(0)
#tf.config.threading.set_inter_op_parallelism_threads(0)
#with tf.device('GPU:2'): # To define the specific GPU to be used.
    model = Sequential()
    # Load the class that will indicate that the initial weight values are desired
    # to be zeros.
    initializer = Zeros()
    # Create the artificial neuron input layer
    model.add(Dense(nodes, input_dim=m, activation='linear', kernel_initializer=initializer))
    # Indicate desired learning rate
    sgd = SGD(learning_rate=0.0001)
    # Compile the keras model
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse'])
    """
    # Define a stop function in which you want the argument variable "loss" of the
    # class "model.compile()" to be monitored and indicate through the variable
    # argument of "EarlyStopping()" named "patience", the number of epochs that
    # you want the training to be stoped if no changes occur in the monitored
    # variable.
    callback = EarlyStopping(monitor='loss', patience=3)
    """
    # fit the keras model on the dataset
    # NOTE: The argument variable "batch_size" represents the desired value that
    #       we want the model to consider before the model updates its weights.
    # NOTE: The argument variable "verbose" indicates if the user wants Keras
    #       to display training messages progress in the terminal window (with 1)
    #       or not (with 0).    
    model.fit(X, Y, epochs=30863, batch_size=n, verbose=0)
    #model.fit(X, Y, epochs=50000, batch_size=n, verbose=0, callbacks=callback)
elapsedTime = time.time() - startingTime
print("Model training with the tensorflow library elapsed " + format(elapsedTime) + " seconds.")
print("")


# ------------------------------------------- #
# ----- Storage of the results obtained ----- #
# ------------------------------------------- #
# We obtained the predicted results by the model that was constructed.
print("Innitializing predictions of the model obtained ...")
startingTime = time.time()
Y_hat = model.predict(X)
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
b = np.zeros((1, m+1))
b[0][1] = model.get_weights()[0][0][0]
b[0][0] = model.get_weights()[1][0]
print("b_0 = " + format(b[0][0]))
print("b_1 = " + format(b[0][1]))

# Compare the results from the CenyML Lybrary and the ones obtained in python.
print("The coefficients will begin their comparation process...")
epsilon = 1e-6
isMatch = 1
for currentRow in range(0, len(idealCoefficients)):
    differentiation = abs(idealCoefficients[currentRow][0] - b[0][currentRow])
    if (differentiation > epsilon):
        isMatch = 0
        print("The absolute differentiation of the coefficient b_: " + format(currentRow) + " exceeded the value defined for epsilon.")
        print("")
        print("The absolute differentiation obtained was: " + format(differentiation))
        break
if (isMatch == 1):
    print("The coefficients obtained in the Python Library matched the ideal coefficient values !!!.")
    
print("The program has finished successfully.")