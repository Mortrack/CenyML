
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: César Miranda Meza
# COMPLETITION DATE: November 09, 2021.
# LAST UPDATE: November 27, 2021.
#
# This code is used to apply the regression evaluation metric known as the
# mean squared error. This is done with the two databases for linear
# equation systems, that differ only because one has a random bias value and
# the other does not. In addition, both of these databases have 1'000'000
# samples each. Moreover, the well known scikit-learn library will be used to
# calculate the mean squared error metric (https://bit.ly/3D20P2A) and then
# its result will be compared with the one obtained with the CenyML library as
# a means of validating the code of CenyML.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Python version 3.9.7


# ----------------------------------- #
# ----- Importing the Libraries ----- #
# ----------------------------------- #
import pandas as pd  # version 1.3.3
import numpy as np # version 1.21.2
import time
from sklearn.metrics import mean_squared_error # version 1.0.1

# -------------------------------------------- #
# ----- Define the user variables values ----- #
# -------------------------------------------- #
m = 1 # This variable is used to define the number of independent variables
      # that the system under study has.
columnIndexOfOutputDataInCsvFile = 2; # This variable will contain the index
                                      # of the first column in which we will
                                      # specify the location of the output
                                      # values (Y and/or Y_hat).

# ------------------------------ #
# ----- Import the dataset ----- #
# ------------------------------ #
# Read the .csv file containing the results of the CenyML library.
print("Innitializing data extraction from .csv file containing the CenyML results ...")
startingTime = time.time()
dataset_CenyML_getMeanSquaredErrorResults = pd.read_csv('CenyML_getMeanSquaredError_Results.csv')
p = len(dataset_CenyML_getMeanSquaredErrorResults.iloc[0])
elapsedTime = time.time() - startingTime
print("Data extraction from .csv file with the CenyML results elapsed " + format(elapsedTime) + " seconds.")
print("")
# Read the .csv file containing the predicted output data.
print("Innitializing data extraction from .csv file containing the predicted output data ...")
startingTime = time.time()
dataset_lES1000S1000SPS = pd.read_csv("../../../../databases/regression/linearEquationSystem/1000systems_1000samplesPerSys.csv")
elapsedTime = time.time() - startingTime
n = len(dataset_lES1000S1000SPS)
csvColumns = len(dataset_lES1000S1000SPS.iloc[0])
print("Data extraction from .csv file containing " + format(n) + " samples for each of the " + format(csvColumns) + " columns (total samples = " + format(n*csvColumns) + ") elapsed " + format(elapsedTime) + " seconds.")
print("")
# Read the .csv file containing the real output data.
print("Innitializing data extraction from .csv file containing the real output data ...")
startingTime = time.time()
dataset_rLES1000S1000SPS = pd.read_csv("../../../../databases/regression/randLinearEquationSystem/1000systems_1000samplesPerSys.csv")
elapsedTime = time.time() - startingTime
n = len(dataset_rLES1000S1000SPS)
csvColumns = len(dataset_rLES1000S1000SPS.iloc[0])
print("Data extraction from .csv file containing " + format(n) + " samples for each of the " + format(csvColumns) + " columns (total samples = " + format(n*csvColumns) + ") elapsed " + format(elapsedTime) + " seconds.")
print("")

# ------------------------------------- #
# ----- Preprocessing of the data ----- #
# ------------------------------------- #
# Retrieving the real data of its corresponding dataset
print("Innitializing real output data with " + format(n) + " samples for each of the " + format(p) + " columns (total samples = " + format(n*p) + ") ...")
startingTime = time.time()
Y = np.zeros((n, 0))
for currentColumn in range(0, p):
    temporalRow = dataset_rLES1000S1000SPS.iloc[:,(currentColumn + columnIndexOfOutputDataInCsvFile)].values.reshape(n, 1)
    Y = np.append(Y, temporalRow, axis=1)
elapsedTime = time.time() - startingTime
print("Real output data innitialization elapsed " + format(elapsedTime) + " seconds.")
print("")
# Retrieving the predicted data of its corresponding dataset
print("Innitializing predicted output data with " + format(n) + " samples for each of the " + format(p) + " columns (total samples = " + format(n*p) + ") ...")
startingTime = time.time()
Y_hat = np.zeros((n, 0))
for currentColumn in range(0, p):
    temporalRow = dataset_lES1000S1000SPS.iloc[:,(currentColumn + columnIndexOfOutputDataInCsvFile)].values.reshape(n, 1)
    Y_hat = np.append(Y_hat, temporalRow, axis=1)
elapsedTime = time.time() - startingTime
print("Predicted output data innitialization elapsed " + format(elapsedTime) + " seconds.")
print("")

# ----------------------------------------------- #
# ----- Apply the mean squared error metric ----- #
# ----------------------------------------------- #
print("Innitializing scikit-learn mean squared error metric calculation ...")
startingTime = time.time()
MSE = mean_squared_error(Y, Y_hat) # We apply the desired evaluation metric.
elapsedTime = time.time() - startingTime
print("scikit-learn mean squared error metric elapsed " + format(elapsedTime) + " seconds.")
print("")

# ---------------------------------------------------------------- #
# ----- Determine if the CenyML Library's method was correct ----- #
# ---------------------------------------------------------------- #
# Compare the results from the CenyML Lybrary and the ones obtained in python.
print("The results will begin their comparation process...")
startingTime = time.time()
epsilon = 3.66e-7
isMatch = 1
for currentColumn in range(0, p):
    differentiation = abs(dataset_CenyML_getMeanSquaredErrorResults.iloc[0][currentColumn] - MSE)
    if (differentiation > epsilon):
        isMatch = 0
        print("The absolute differentiation of the Column: " + dataset_CenyML_getMeanSquaredErrorResults.columns.tolist()[currentColumn] + " and the Row: " + format(0) + " exceeded the value defined for epsilon.")
        print("The absolute differentiation obtained was: " + format(differentiation))
        break
if (isMatch == 1):
    print("The results obtained in Python and in the CenyML Library matched !!!.")
elapsedTime = time.time() - startingTime
print("The comparation process elapsed " + format(elapsedTime) + " seconds.")

