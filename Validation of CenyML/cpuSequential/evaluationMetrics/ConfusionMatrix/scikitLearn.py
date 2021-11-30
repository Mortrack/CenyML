
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: César Miranda Meza
# COMPLETITION DATE: November 23, 2021.
# LAST UPDATE: November 27, 2021.
#
# This code is used to apply the classification evaluation metric known as the
# confusion matrix. This is done with the two databases for linear equation
# systems, that differ only because one has a random bias value and the other
# does not. In addition, both of these databases have 1'000'000 samples each.
# Moreover, the well known scikit-learn library will be used to
# calculate the confusion matrix metric (https://bit.ly/3cHJws2) and then its
# result will be compared with the one obtained with the CenyML library as a
# means of validating the code of CenyML.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Python version 3.9.7


# ----------------------------------- #
# ----- Importing the Libraries ----- #
# ----------------------------------- #
import pandas as pd  # version 1.3.3
import numpy as np # version 1.21.2
import time
from sklearn.metrics import confusion_matrix # version 1.0.1

# -------------------------------------------- #
# ----- Define the user variables values ----- #
# -------------------------------------------- #
m = 1 # This variable is used to define the number of independent variables
      # that the system under study has.
p = 1 # This variable is used to define the number of dependent variables
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
dataset_CenyML_getConfusionMatrixResults = pd.read_csv('CenyML_getConfusionMatrix_Results.csv')
elapsedTime = time.time() - startingTime
print("Data extraction from .csv file with the CenyML results elapsed " + format(elapsedTime) + " seconds.")
print("")

# Read the .csv file containing the real output data.
print("Innitializing data extraction from .csv file containing the real output data ...")
startingTime = time.time()
dataset_rLES1000S1000SPS = pd.read_csv("../../../../databases/classification/randLinearEquationSystem/100systems_100samplesPerAxisPerSys.csv")
elapsedTime = time.time() - startingTime
n = len(dataset_rLES1000S1000SPS)
csvColumns = len(dataset_rLES1000S1000SPS.iloc[0])
print("Data extraction from .csv file containing " + format(n) + " samples for each of the " + format(csvColumns) + " columns (total samples = " + format(n*csvColumns) + ") elapsed " + format(elapsedTime) + " seconds.")
print("")

# Read the .csv file containing the predicted output data.
print("Innitializing data extraction from .csv file containing the predicted output data ...")
startingTime = time.time()
dataset_lES1000S1000SPS = pd.read_csv("../../../../databases/classification/linearEquationSystem/100systems_100samplesPerAxisPerSys.csv")
elapsedTime = time.time() - startingTime
n = len(dataset_lES1000S1000SPS)
csvColumns = len(dataset_lES1000S1000SPS.iloc[0])
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


# --------------------------------------------- #
# ----- Apply the confusion matrix metric ----- #
# --------------------------------------------- #
print("Innitializing scikit-learn confusion matrix metric calculation ...")
startingTime = time.time()
confusionMatrix = confusion_matrix(Y, Y_hat).ravel()
elapsedTime = time.time() - startingTime
print("scikit-learn confusion matrix metric elapsed " + format(elapsedTime) + " seconds.")
print("")
# Swap the values contained in the first and last indexes of the result
# obtained so that they are arranged in the same manner as in the CenyML
# library.
tmp = confusionMatrix[0]
confusionMatrix[0] = confusionMatrix[3]
confusionMatrix[3] = tmp

# ---------------------------------------------------------------- #
# ----- Determine if the CenyML Library's method was correct ----- #
# ---------------------------------------------------------------- #
# Compare the results from the CenyML Lybrary and the ones obtained in python.
print("The results will begin their comparation process...")
startingTime = time.time()
epsilon = 1.0e-6
isMatch = 1
for currentColumn in range(0, 4):
    differentiation = abs(dataset_CenyML_getConfusionMatrixResults.iloc[0][currentColumn] - confusionMatrix[currentColumn])
    if (differentiation > epsilon):
        isMatch = 0
        print("The absolute differentiation of the Column: " + dataset_CenyML_getConfusionMatrixResults.columns.tolist()[currentColumn] + " and the Row: " + format(0) + " exceeded the value defined for epsilon.")
        print("The absolute differentiation obtained was: " + format(differentiation))
        break
if (isMatch == 1):
    print("The results obtained in Python and in the CenyML Library matched !!!.")
elapsedTime = time.time() - startingTime
print("The comparation process elapsed " + format(elapsedTime) + " seconds.")

