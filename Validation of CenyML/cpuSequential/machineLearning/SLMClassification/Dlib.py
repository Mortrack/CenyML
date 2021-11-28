
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: César Miranda Meza
# COMPLETITION DATE: November 24, 2021.
# LAST UPDATE: November 27, 2021.
#
# This code is used to apply the machine learning method known as the linear
# support vector machine classification. This is done with the database used
# for linear equation systems. In addition, this database has 1'000'000
# samples. Moreover, the well known Dlib library will be used for such machine
# learning algorithm (http://dlib.net/). Then, some metrics will be obtained,
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
from sklearn.metrics import log_loss # version 1.0.1
from sklearn.metrics import confusion_matrix # version 1.0.1
from sklearn.metrics import accuracy_score # version 1.0.1
from sklearn.metrics import precision_score # version 1.0.1
from sklearn.metrics import recall_score # version 1.0.1
from sklearn.metrics import f1_score # version 1.0.1
import dlib # version 19.22
try:
    import cPickle as pickle
except ImportError:
    import pickle


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
X_Dlib = dlib.vectors()
Y_Dlib = dlib.array()
for currentRow in range(0, n):
    X_Dlib.append(dlib.vector([X[currentRow][0], X[currentRow][1]]))
for currentRow in range(0, n):
    temporalValue = dataset_lES100S100SPAPS.iloc[currentRow, (columnIndexOfOutputDataInCsvFile)]
    if temporalValue == 0:
        temporalValue = -1
    Y_Dlib.append(temporalValue)
Y_ravel = np.ravel(Y)
elapsedTime = time.time() - startingTime
print("Input and output data innitialization elapsed " + format(elapsedTime) + " seconds.")
print("")


# -------------------------- #
# ----- Model training ----- #
# -------------------------- #
print("Innitializing model training with the Dlib library ...")
startingTime = time.time()
# Now make a training object.  This object is responsible for turning a
# training dataset into a prediction model.  This one here is a SVM trainer
# that uses a linear kernel.  If you wanted to use a RBF kernel or histogram
# intersection kernel you could change it to one of these lines:
#  svm = dlib.svm_c_trainer_histogram_intersection()
#  svm = dlib.svm_c_trainer_radial_basis()
svm = dlib.svm_c_trainer_linear()
svm.be_verbose()
svm.set_c(1000)
# Now train the model.  The return value is the trained model capable of making predictions.
classifier = svm.train(X_Dlib, Y_Dlib)
elapsedTime = time.time() - startingTime
print("Model training with the Dlib library elapsed " + format(elapsedTime) + " seconds.")
print("")


# ------------------------------------------- #
# ----- Storage of the results obtained ----- #
# ------------------------------------------- #
# We obtained the predicted results by the model that was constructed.
print("Innitializing predictions of the model obtained ...")
startingTime = time.time()
Y_hat = []
for currentRow in range(0, n):
    if (classifier(X_Dlib[currentRow]) > 0):
        Y_hat.append([1])
    else:
        Y_hat.append([0])
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

# Save the Dlib machine learning model that was generated
with open('saved_model(Dlib).pickle', 'wb') as handle:
    pickle.dump(classifier, handle, 2)

