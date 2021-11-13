
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: Cesar Miranda Meza
# COMPLETITION DATE: September 22, 2021.
# LAST UPDATE: November 08, 2021
#
# This code is used for obtain the mean of each of the columns contained in the
# database "multiplePolynomialEquationSystem_100systems_100samplesPerAxisPerSys",
# which has 1'000'000 samples for for each of them (making a total of 5'000'000).
# For this purpose, the well known NumPy library will be used to calculate the
# mean (https://github.com/numpy/numpy) and then it will be compared with the
# results that were obtained with the CenyML Library as a means of validating
# that the code created in this library for the mean method is correct.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


# ----------------------------------- #
# ----- Importing the Libraries ----- #
# ----------------------------------- #
import pandas as pd  # version 1.3.3
import numpy as np # version 1.21.2
import time

# ------------------------------ #
# ----- Import the dataset ----- #
# ------------------------------ #
# Read the .csv file containing the results of the CenyML library.
print("Innitializing data extraction from .csv file containing the CenyML results ...")
startingTime = time.time()
dataset_CenyML_meanResults = pd.read_csv('CenyML_getMean_Results.csv')
elapsedTime = time.time() - startingTime
print("Data extraction from .csv file with the CenyML results elapsed " + format(elapsedTime) + " seconds.")
print("")
# Read the .csv file containing the reference data.
print("Innitializing data extraction from .csv file containing the reference input data ...")
startingTime = time.time()
dataset_mPES100S100SPAPS = pd.read_csv('../../../../Databases/regressionDBs/randMultiplePolynomialEquationSystem/100systems_100samplesPerAxisPerSys.csv')
elapsedTime = time.time() - startingTime
n = len(dataset_mPES100S100SPAPS)
m = len(dataset_mPES100S100SPAPS.iloc[0])
desired_m = len(dataset_CenyML_meanResults.iloc[0])
print("Data extraction from .csv file containing " + format(n) + " samples for each of the " + format(m) + " columns (total samples = " + format(n*m) + ") elapsed " + format(elapsedTime) + " seconds.")
print("")

# ------------------------------------- #
# ----- Preprocessing of the data ----- #
# ------------------------------------- #
print("Innitializing input data with " + format(n) + " samples for each of the " + format(desired_m) + " columns (total samples = " + format(n*desired_m) + ") ...")
startingTime = time.time()
X = np.zeros((n, 0))
for currentColumn in range(0, int(desired_m/m)):
    for currentColumnCsv in range(0, m):
        temporalRow = dataset_mPES100S100SPAPS.iloc[:,currentColumnCsv].values.reshape(n, 1)
        X = np.append(X, temporalRow, axis=1)
elapsedTime = time.time() - startingTime
print("Input data innitialization elapsed " + format(elapsedTime) + " seconds.")
print("")

# ------------------------------ #
# ----- Calculate the mean ----- #
# ------------------------------ #
print("Innitializing NumPy mean method calculation ...")
startingTime = time.time()
means = np.mean(X, axis=0)
elapsedTime = time.time() - startingTime
print("NumPy mean method elapsed " + format(elapsedTime) + " seconds.")
print("")

# ---------------------------------------------------------------- #
# ----- Determine if the CenyML Library's method was correct ----- #
# ---------------------------------------------------------------- #
# Compare the results from the CenyML Lybrary and the ones obtained in python.
print("The results will begin their comparation process...")
epsilon = 1e-7
isMatch = 1
for currentColumn in range(0, len(dataset_mPES100S100SPAPS.iloc[0]) ):
    differentiation = abs(dataset_CenyML_meanResults.iloc[0][currentColumn] - means[currentColumn])
    if (differentiation > epsilon):
        isMatch = 0
        print("The absolute differentiation of the Columns: " + dataset_mPES100S100SPAPS.columns.tolist()[currentColumn] + " exceeded the value defined for epsilon.")
        print("")
        print("The absolute differentiation obtained was: " + format(differentiation))
        break
if (isMatch == 1):
    print("The results obtained in Python and in the CenyML Library matched !!!.")

