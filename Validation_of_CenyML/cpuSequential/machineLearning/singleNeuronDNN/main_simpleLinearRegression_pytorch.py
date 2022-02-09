# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: César Miranda Meza
# COMPLETITION DATE: December 27, 2021.
# LAST UPDATE: N/A.
#
# This code is used to apply the machine learning method known as artificial
# neural network but with only a single neuron to solve a simple linear
# regression problem. This is done with the database used for linear equation
# systems. In addition, this database has 1'000'000 samples. Moreover, the
# well known PyTorch library will be used for such machine learning algorithm
# (https://bit.ly/3Jm0nzI, https://bit.ly/33Sjr8w and https://bit.ly/3GTz1Q9).
# Then, some metrics will be obtained, along with a plot to use these as a
# comparative evaluation of the results obtained in the CenyML library.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Python version 3.9.7


# ----------------------------------- #
# ----- Importing the Libraries ----- #
# ----------------------------------- #
import pandas as pd  # version 1.3.3
import numpy as np # version 1.21.2
import time
import torch # version 1.10.1
import torch.nn as nn # version 1.10.1
from sklearn.metrics import mean_squared_error # version 1.0.1
import matplotlib.pyplot as plt # version 3.4.3


# -------------------------------------------- #
# ----- Define the user variables values ----- #
# -------------------------------------------- #
m = 1 # This variable is used to define the number of independent variables
      # that the system under study has.
p = 1 # This variable is used to define the number of dependent variables
      # that the system under study has.
learning_rate = 0.00001 # This variable is used to define the desired
                       # learning rate for the model to be trained.
num_epochs = 30863 # This variable is used to define the desired number of
                   # epochs for the model to be trained.
desiredNumberOfCpuThreads = 1 # This variable is used to define the desired
                              # number of CPU threads that wants to be used
                              # during the training of the deep learning model.
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
# 0) Prepare data for Pytorch model
# Cast to float Tensor
X_torch = torch.from_numpy(X.astype(np.float32))
Y_torch = torch.from_numpy(Y.astype(np.float32))
Y_torch = Y_torch.view(Y_torch.shape[0], 1)
n_samples, n_features = X_torch.shape
elapsedTime = time.time() - startingTime
print("Input and output data innitialization elapsed " + format(elapsedTime) + " seconds.")
print("")


# -------------------------- #
# ----- Model training ----- #
# -------------------------- #
print("Innitializing model training with the PyTorch library ...")
startingTime = time.time()
# 1) Define the number of CPU threads to be used
torch.set_num_threads(desiredNumberOfCpuThreads)
# 2) Model
# Linear model f = wx + b
model = nn.Linear(m, p) # Input and output equal 1 so that only a single neuron is trained.
# Define initial weight values with zeros
zeroValue_torch = torch.zeros(1,1)
model.bias.data = zeroValue_torch[0]
model.weight.data = zeroValue_torch
# 3) Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
# 4) Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()
    Y_predicted = model(X_torch)
    loss = criterion(Y_predicted, Y_torch)
    loss.backward()
    optimizer.step()
elapsedTime = time.time() - startingTime
print("Model training with the PyTorch library elapsed " + format(elapsedTime) + " seconds.")
print("")


# ------------------------------------------- #
# ----- Storage of the results obtained ----- #
# ------------------------------------------- #
# We obtained the predicted results by the model that was constructed.
print("Innitializing predictions of the model obtained ...")
startingTime = time.time()
Y_hat = model(X_torch).detach().numpy()
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
plt.scatter(X_torch, Y, color='red')
plt.plot(X_torch, Y_hat, color='blue')
plt.title('Simple linear regression with the \"Sequential()\" neural network of tensorflow')
plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')

# We display, in console, the coefficient values obtained with the ML method used.
# NOTE: The bias and weight values will be stored in "b" such that each column
#       will contain the bias and weight values of a certain neuron.
b = np.zeros((1, m+1))
# NOTE: ".detach().numpy()" converts a tensor to numpy value
b[0][0] = model.bias[0].detach().numpy()
b[0][1] = model.weight[0][0].detach().numpy()
print("b_0 = " + format(b[0][0]))
print("b_1 = " + format(b[0][1]))

# Compare the ideal results with the ones obtained in python.
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