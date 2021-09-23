
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: Cesar Miranda Meza
# COMPLETITION DATE: September 22, 2020.
# LAST UPDATE: N/A
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
import pandas as pd
import numpy as np

# ------------------------------ #
# ----- Import the dataset ----- #
# ------------------------------ #
dataset_mPES100S100SPAPS = pd.read_csv('../../Databases/regressionDBs/multiplePolynomialEquationSystem/multiplePolynomialEquationSystem_100systems_100samplesPerAxisPerSys.csv')
print("The following will give an insight of the contents of the database that has been loaded:")
print(dataset_mPES100S100SPAPS.iloc[:,:])
print("")

# ------------------------------ #
# ----- Calculate the mean ----- #
# ------------------------------ #
means = np.mean( dataset_mPES100S100SPAPS)

# ------------------------------------ #
# ----- Display results obtained ----- #
# ------------------------------------ #
print("The results obtained in Python are the following:")
for currentColumn in range(0, len(dataset_mPES100S100SPAPS.iloc[0]) ):
    print("The mean of the column " + format(currentColumn) + " is: " + format(means[currentColumn]))    
print("")

# ---------------------------------------------------------------- #
# ----- Determine if the CenyML Library's method was correct ----- #
# ---------------------------------------------------------------- #
# Read the .csv file containing the results of the CenyML library.
dataset_CenyML_meanResults = pd.read_csv('CenyML_getMean_Results.csv')
# Compare the results from the CenyML Lybrary and the ones obtained in python.
print("The results will begin their comparation process...")
epsilon = 1e-7
isMatch = 1
for currentColumn in range(0, len(dataset_mPES100S100SPAPS.iloc[0]) ):
    differentiation = abs(dataset_CenyML_meanResults.iloc[0][currentColumn] - means[currentColumn])
    if (differentiation > epsilon):
        isMatch = 0
        print("The absolute differentiation of the Columns: " + dataset_mPES100S100SPAPS.columns.tolist()[0] + " exceeded the value defined for epsilon.")
        break
if (isMatch == 1):
    print("The results obtained in Python and in the CenyML Library matched !!!.")

