
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: César Miranda Meza
# COMPLETITION DATE: November 12, 2021.
# LAST UPDATE: November 23, 2021.
#
# This code is used to apply the regression evaluation metric known as the
# adjusted coefficient of determination. This is done with the database used
# for Multiple linear equation systems that contains a random bias value. In
# addition, both of this database has 1'000'000 samples. Moreover, the well
# known statsmodels library will be used to calculate the adjusted coefficient
# of determination metric (https://bit.ly/31RbejT) but by first applying a
# linear regression since it is requested before being able to apply such
# metric. Finally, two .csv files will be created where one will store the
# result obtained by such metric and the other .csv file will store the
# predicted values that were used for it. These two databases will be used in
# the C file for the validation of the adjusted coefficient of determination
# metric.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Python version 3.9.7


# ----------------------------------- #
# ----- Importing the Libraries ----- #
# ----------------------------------- #
import pandas as pd  # version 1.3.3
import numpy as np # version 1.21.2
import time
import statsmodels.api as sm # version 0.13.0

# -------------------------------------------- #
# ----- Define the user variables values ----- #
# -------------------------------------------- #
m = 2 # This variable is used to define the number of independent variables
      # that the system under study has.
p = 1 # This variable is used to define the number of dependent variables
      # that the system under study has.
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
# Read the .csv file containing the real output data.
print("Innitializing data extraction from .csv file containing the real output data ...")
startingTime = time.time()
dataset_rMLES100S100SPAPS = pd.read_csv("../../../../databases/regressionDBs/randMultipleLinearSystem/100systems_100samplesPerAxisPerSys.csv")
elapsedTime = time.time() - startingTime
n = len(dataset_rMLES100S100SPAPS)
csvColumns = len(dataset_rMLES100S100SPAPS.iloc[0])
print("Data extraction from .csv file containing " + format(n) + " samples for each of the " + format(csvColumns) + " columns (total samples = " + format(n*csvColumns) + ") elapsed " + format(elapsedTime) + " seconds.")
print("")

# ------------------------------------- #
# ----- Preprocessing of the data ----- #
# ------------------------------------- #
# Retrieving the real data of its corresponding dataset
print("Innitializing input and output data with " + format(n) + " samples for each of the " + format(p) + " columns (total samples = " + format(n*p) + ") ...")
startingTime = time.time()
X, Y = dataset_rMLES100S100SPAPS[["independent_variable_1", "independent_variable_2"]], dataset_rMLES100S100SPAPS.dependent_variable
elapsedTime = time.time() - startingTime
print("Input and output data innitialization elapsed " + format(elapsedTime) + " seconds.")
print("")

# -------------------------- #
# ----- Model training ----- #
# -------------------------- #
print("Innitializing model training with the statsmodels library ...")
startingTime = time.time()
X = sm.add_constant(X)
model = sm.OLS(Y, X).fit()
elapsedTime = time.time() - startingTime
print("Model training with the statsmodels library elapsed " + format(elapsedTime) + " seconds.")
print("")

# ------------------------------------------------------------------ #
# ----- Apply the adjusted coefficient of determination metric ----- #
# ------------------------------------------------------------------ #
print("Innitializing statsmodels adjusted coefficient of determination metric calculation ...")
startingTime = time.time()
adjustedRsquared = model.rsquared_adj
elapsedTime = time.time() - startingTime
print("statsmodels adjusted coefficient of determination metric elapsed " + format(elapsedTime) + " seconds.")
print("")

# ------------------------------------------- #
# ----- Storage of the results obtained ----- #
# ------------------------------------------- #
# We save the obtained results in a .csv file
print("Innitializing the creation of a new .csv file to store the results obtained ...")
startingTime = time.time()
f = open("adjustedRsquared_results.csv","w+") # Create a new file to store data in it.
f.write("adjustedRsquaredResults\n") # Write the header titles for each of the desired columns.
f.write(format(adjustedRsquared)) # Write the result obtained.
f.close() # Close the file to conclude its creation.
elapsedTime = time.time() - startingTime
print("Creation of a new .csv file to store the results obtained elapsed " + format(elapsedTime) + " seconds.")
print("")

# We save the obtained predicted outputs in a .csv file
print("Innitializing the creation of a new .csv file to store the predicted data obtained ...")
startingTime = time.time()
Y_hat = model.predict(X) # Obtain the predicted data that was used to calculate the adjusted coefficient of determination.
f = open("adjustedRsquared_predictedData.csv","w+") # Create a new file to store data in it.
f.write("predictedData\n") # Write the header titles for each of the desired columns.
for currentSampleOfCurrentSystem in range(0, n):
    f.write(format(Y_hat[currentSampleOfCurrentSystem]) + "\n") # Write the new line of data into the file.
f.close() # Close the file to conclude its creation.
elapsedTime = time.time() - startingTime
print("Creation of a new .csv file to store the predicted data obtained elapsed " + format(elapsedTime) + " seconds.")
print("")

