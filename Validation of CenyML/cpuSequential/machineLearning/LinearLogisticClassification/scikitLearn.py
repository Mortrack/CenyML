
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: César Miranda Meza
# COMPLETITION DATE: November 24, 2021.
# LAST UPDATE: November 27, 2021.
#
# This code is used to apply the machine learning method known as the linear
# logistic classification. This is done with the database used for linear
# equation systems. In addition, this database has 1'000'000 samples. Moreover,
# the well known scikit-learn library will be used for such machine learning
# algorithm (https://bit.ly/3FMEXsT). Then, some metrics will be obtained,
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
from sklearn.linear_model import LogisticRegression # version 1.0.1
from sklearn.metrics import log_loss # version 1.0.1
from sklearn.metrics import confusion_matrix # version 1.0.1
from sklearn.metrics import accuracy_score # version 1.0.1
from sklearn.metrics import precision_score # version 1.0.1
from sklearn.metrics import recall_score # version 1.0.1
from sklearn.metrics import f1_score # version 1.0.1
from matplotlib.colors import ListedColormap # version 3.4.3
import matplotlib.pyplot as plt # version 3.4.3


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
# Read the .csv file containing the data to be trained with.
print("Innitializing data extraction from .csv file containing the data to train with ...")
startingTime = time.time()
dataset_lES100S100SPAPS = pd.read_csv("../../../../databases/classificationDBs/linearClassificationSystem/100systems_100samplesPerAxisPerSys.csv")
elapsedTime = time.time() - startingTime
n = len(dataset_lES100S100SPAPS)
csvColumns = len(dataset_lES100S100SPAPS.iloc[0])
print("Data extraction from .csv file containing " + format(n) + " samples for each of the " + format(csvColumns) + " columns (total samples = " + format(n*csvColumns) + ") elapsed " + format(elapsedTime) + " seconds.")
print("")


# ------------------------------------- #
# ----- Preprocessing of the data ----- #
# ------------------------------------- #
# Retrieving the real data of its corresponding dataset
print("Innitializing input and output data with " + format(n) + " samples for each of the " + format(p) + " columns (total samples = " + format(n*p) + ") ...")
startingTime = time.time()
Y = np.ones((n, 0))
X = np.ones((n, 0))
for currentColumn in range(0, p):
    temporalRow = dataset_lES100S100SPAPS.iloc[:,currentColumn + columnIndexOfOutputDataInCsvFile].values.reshape(n, 1)
    Y = np.append(Y, temporalRow, axis=1)
for currentColumn in range(0, m):
    temporalRow = dataset_lES100S100SPAPS.iloc[:,currentColumn + columnIndexOfInputDataInCsvFile].values.reshape(n, 1)
    X = np.append(X, temporalRow, axis=1)
Y_ravel = np.ravel(Y)
elapsedTime = time.time() - startingTime
print("Input and output data innitialization elapsed " + format(elapsedTime) + " seconds.")
print("")


# -------------------------- #
# ----- Model training ----- #
# -------------------------- #
print("Innitializing model training with the scikit-learn library ...")
startingTime = time.time()
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, Y_ravel)
elapsedTime = time.time() - startingTime
print("Model training with the scikit-learn library elapsed " + format(elapsedTime) + " seconds.")
print("")


# ------------------------------------------- #
# ----- Storage of the results obtained ----- #
# ------------------------------------------- #
# We obtained the predicted results by the model that was constructed.
print("Innitializing predictions of the model obtained ...")
startingTime = time.time()
Y_hat = classifier.predict(X)
elapsedTime = time.time() - startingTime
print("Model predictions elapsed " + format(elapsedTime) + " seconds.")
print("")

# We obtain the cross entropy error metric on the obtained ML model.
print("Innitializing scikit-learn cross entropy error metric calculation ...")
startingTime = time.time()
NLL = log_loss(Y, Y_hat, normalize=False) # We apply the desired evaluation metric.
elapsedTime = time.time() - startingTime
print("The result obtained was NLL = " + format(NLL) )
print("scikit-learn cross entropy error metric elapsed " + format(elapsedTime) + " seconds.")
print("")

# We obtain the confusion matrix metric on the obtained ML model.
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

# We obtain the accuracy metric on the obtained ML model.
print("Innitializing scikit-learn accuracy metric calculation ...")
startingTime = time.time()
accuracy = accuracy_score(Y, Y_hat)
elapsedTime = time.time() - startingTime
print("scikit-learn accuracy metric elapsed " + format(elapsedTime) + " seconds.")
print("")

# We obtain the precision metric on the obtained ML model.
print("Innitializing scikit-learn precision metric calculation ...")
startingTime = time.time()
precision = precision_score(Y, Y_hat)
elapsedTime = time.time() - startingTime
print("scikit-learn precision metric elapsed " + format(elapsedTime) + " seconds.")
print("")

# We obtain the recall metric on the obtained ML model.
print("Innitializing scikit-learn recall metric calculation ...")
startingTime = time.time()
recall = recall_score(Y, Y_hat)
elapsedTime = time.time() - startingTime
print("scikit-learn recall metric elapsed " + format(elapsedTime) + " seconds.")
print("")

# We obtain the F1 score metric on the obtained ML model.
print("Innitializing scikit-learn F1 score metric calculation ...")
startingTime = time.time()
F1score = f1_score(Y, Y_hat)
elapsedTime = time.time() - startingTime
print("scikit-learn F1 score metric elapsed " + format(elapsedTime) + " seconds.")
print("")

# We visualize the trainning set results
X_set, Y_set = X, np.reshape(Y, (n))
rangeToBeAdded1 = (X_set[:, 0].max() - X_set[:, 0].min()) * 0.5
rangeToBeAdded2 = (X_set[:, 1].max() - X_set[:, 1].min()) * 0.5
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - rangeToBeAdded1, stop = X_set[:, 0].max() + rangeToBeAdded1, step = 0.1),
                     np.arange(start = X_set[:, 1].min() - rangeToBeAdded2, stop = X_set[:, 1].max() + rangeToBeAdded2, step = 0.1))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.50, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Linear logistic classification')
plt.xlabel('Independent variable 1')
plt.ylabel('Independent variable 2')
plt.legend()
plt.show()
