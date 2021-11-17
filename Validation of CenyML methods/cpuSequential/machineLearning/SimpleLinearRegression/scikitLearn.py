
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: César Miranda Meza
# COMPLETITION DATE: November 13, 2021.
# LAST UPDATE: November 17, 2021.
#
# This code is used to apply the machine learning method known as the simple
# linear regression. This is done with the database used for linear equation
# systems. In addition, this database has 1'000'000 samples. Moreover, the
# well known scikit-learn library will be used for such machine learning
# algorithm (https://bit.ly/3FghUqa). Then, some metrics will be obtained,
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
from sklearn.linear_model import LinearRegression # version 1.0.1
from sklearn.metrics import mean_squared_error # version 1.0.1
import matplotlib.pyplot as plt # version 3.4.3


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
columnIndexOfInputDataInCsvFile = 3; # This variable will contain the index
                                     # of the first column in which we will
                                     # specify the location of the input
                                     # values (X).
                                     
                                     
# ------------------------------ #
# ----- Import the dataset ----- #
# ------------------------------ #
# Read the .csv file containing the results of the CenyML library.
print("Innitializing data extraction from .csv file containing the CenyML coefficient results ...")
startingTime = time.time()
dataset_CenyML_linearRegresCoeff = pd.read_csv('CenyML_getSimpleLinearRegression_Coefficients.csv')
elapsedTime = time.time() - startingTime
print("Data extraction from .csv file with the CenyML coefficient results elapsed " + format(elapsedTime) + " seconds.")
print("")

# Read the .csv file containing the data to be trained with.
print("Innitializing data extraction from .csv file containing the data to train with ...")
startingTime = time.time()
dataset_rLES1000S1000SPS = pd.read_csv("../../../../Databases/regressionDBs/linearEquationSystem/1000systems_1000samplesPerSys.csv")
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
regressor = LinearRegression()
regressor.fit(X, Y)
b = np.zeros((1, m+1))
b[0][1] = regressor.coef_[0][0]
elapsedTime = time.time() - startingTime
print("Model training with the scikit-learn library elapsed " + format(elapsedTime) + " seconds.")
print("")


# ------------------------------------------- #
# ----- Storage of the results obtained ----- #
# ------------------------------------------- #
# We obtained the predicted results by the model that was constructed.
print("Innitializing predictions of the model obtained ...")
startingTime = time.time()
Y_hat = regressor.predict(X)
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

# We obtained the coefficient of determination metric on the obtained ML model.
print("Innitializing scikit-learn R-squared metric calculation ...")
startingTime = time.time()
Rsquared = regressor.score(X, Y)
elapsedTime = time.time() - startingTime
print("R-squared = " + format(Rsquared))
print("R-squared metric with the scikit-learn library elapsed " + format(elapsedTime) + " seconds.")
print("")

# We visualize the trainning set results
X_plot = dataset_rLES1000S1000SPS.independent_variable_1
plt.scatter(X_plot, Y, color='red')
plt.plot(X_plot, Y_hat, color='blue')
plt.title('Simple linear regression with scikit-learn')
plt.xlabel('Independent variable')
plt.ylabel('Dependent variable')

# We display, in console, the coefficient values obtained with the ML method used.
b[0][0] = Y_hat[0][0] - b[0][1]*X[0][0]
print("b_0 = " + format(b[0][0]))
print("b_1 = " + format(b[0][1]))

# Compare the results from the CenyML Lybrary and the ones obtained in python.
print("The coefficients will begin their comparation process...")
epsilon = 1e-6
isMatch = 1
for currentRow in range(0, len(dataset_CenyML_linearRegresCoeff) ):
    differentiation = abs(dataset_CenyML_linearRegresCoeff.iloc[currentRow][0] - b[0][currentRow])
    if (differentiation > epsilon):
        isMatch = 0
        print("The absolute differentiation of the coefficient b_: " + format(currentRow) + " exceeded the value defined for epsilon.")
        print("")
        print("The absolute differentiation obtained was: " + format(differentiation))
        break
if (isMatch == 1):
    print("The coefficients obtained in Python and in the CenyML Library matched !!!.")