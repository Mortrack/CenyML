
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: Cesar Miranda Meza
# COMPLETITION DATE: October 18, 2021.
# LAST UPDATE: N/A
#
# This code is used for obtain the mode of each of the columns contained in the
# database "polynomialClassificationSystem_10systems_100samplesPerAxisPerSys",
# which has 100'000 samples for each of them (making a total of 500'000).
# For this purpose, the well known statistics library will be used to calculate
# the mode (https://docs.python.org/3/library/statistics.html) and then it will
# be compared with the results that were obtained with the CenyML Library as a
# means of validating that the code created in this library for the mode method
# is correct.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


# ----------------------------------- #
# ----- Importing the Libraries ----- #
# ----------------------------------- #
import pandas as pd
import numpy as np
import statistics
import time

# ------------------------------ #
# ----- Import the dataset ----- #
# ------------------------------ #
# Read the .csv file containing the results of the CenyML library.
print("Innitializing data extraction from .csv file containing the CenyML results ...")
startingTime = time.time()
dataset_CenyML_quickModeResults = pd.read_csv('CenyML_getQuickMode_Results.csv')
elapsedTime = time.time() - startingTime
print("Data extraction from .csv file with the CenyML results elapsed " + format(elapsedTime) + " seconds.")
print("")
# Read the .csv file containing the reference data.
print("Innitializing data extraction from .csv file containing the reference input data ...")
startingTime = time.time()
dataset_pCS10S100SPAPS = pd.read_csv('../../../../Databases/classificationDBs/polynomialClassificationSystem/polynomialClassificationSystem_10systems_100samplesPerAxisPerSys.csv')
elapsedTime = time.time() - startingTime
n = len(dataset_pCS10S100SPAPS)
m = len(dataset_pCS10S100SPAPS.iloc[0])
desired_m = len(dataset_CenyML_quickModeResults.iloc[0])
print("Data extraction from .csv file containing " + format(n) + " samples for each of the " + format(m) + " columns (total samples = " + format(n*m) + ") elapsed " + format(elapsedTime) + " seconds.")
print("")

# ------------------------------------- #
# ----- Preprocessing of the data ----- #
# ------------------------------------- #
print("Innitializing input data with " + format(n) + " samples for each of the " + format(desired_m) + " columns (total samples = " + format(n*desired_m) + ") ...")
startingTime = time.time()
X = []
for currentColumn in range(0, int(desired_m/m)):
    for currentColumnCsv in range(0, m):
        temporalArray = []
        for currentRow in range (0, len(dataset_pCS10S100SPAPS)):
            temporalArray.append(dataset_pCS10S100SPAPS.iloc[currentRow][currentColumnCsv])
        X.append(temporalArray)
elapsedTime = time.time() - startingTime
print("Input data innitialization elapsed " + format(elapsedTime) + " seconds.")
print("")

# ------------------------------ #
# ----- Calculate the mode ----- #
# ------------------------------ #
print("Innitializing statistics mode method calculation ...")
startingTime = time.time()
Mo_n = []
modeValues = []
for currentColumn in range(0, int(desired_m)):
    modeValues.append(statistics.multimode(X[currentColumn]))
    Mo_n.append(len(modeValues[currentColumn]))
elapsedTime = time.time() - startingTime
print("statistics mode method elapsed " + format(elapsedTime) + " seconds.")
print("")

# ---------------------------------------------------------------- #
# ----- Determine if the CenyML Library's method was correct ----- #
# ---------------------------------------------------------------- #
# Compare the results from the CenyML Lybrary and the ones obtained in python.
print("The results will begin their comparation process...")
startingTime = time.time()
epsilon = 1e-20
isMatch = 1
for currentColumn in range(0, len(dataset_pCS10S100SPAPS.iloc[0]) ):
    for currentRow in range(0, Mo_n[currentColumn]):
        differentiation = abs(dataset_CenyML_quickModeResults.iloc[currentRow][currentColumn] - modeValues[currentColumn][currentRow])
        if (differentiation > epsilon):
            isMatch = 0
            print("The absolute differentiation of the Column: " + dataset_pCS10S100SPAPS.columns.tolist()[currentColumn] + " and the Row: " + format(currentRow) + " exceeded the value defined for epsilon.")
            print("")
            print("The absolute differentiation obtained was: " + format(differentiation))
            break
if (isMatch == 1):
    print("The results obtained in Python and in the CenyML Library matched !!!.")
elapsedTime = time.time() - startingTime
print("The comparation process elapsed " + format(elapsedTime) + " seconds.")
