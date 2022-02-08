/*
* This program will load the same data contained in the CenyML .csv
* file that has the data of a linear equation system. Its input data
* will be saved into the matrix "X" and its output data into the
* matrix "Y". Subsequently, a simple linear regression method will be
* applied to obtain the best fitting coefficient values of such data.
* Next, some evaluation metrics will be applied and then a validation
* of the coefficients obtained by the model will be made with respect
* to what is expected.
 */


 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include "CenyML_Library/cpuSequential/statistics/CenyMLstatistics.h" // library to use the statistics methods of CenyML.
#include "CenyML_Library/cpuSequential/featureScaling/CenyMLfeatureScaling.h" // library to use the feature scaling methods of CenyML.
#include "CenyML_Library/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.h" // library to use the regression evaluation metrics of CenyML.
#include "CenyML_Library/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.h" // library to use the classification evaluation metrics of CenyML.
#include "CenyML_Library/cpuSequential/machineLearning/CenyMLregression.h" // library to use the regression algorithms of CenyML.
#include "CenyML_Library/cpuSequential/machineLearning/CenyMLclassification.h" // library to use the classification algorithms of CenyML.
#include "CenyML_Library/cpuSequential/machineLearning/CenyMLdeepLearning.h" // library to use the deep learning algorithms of CenyML.


void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);// initialize serial communications at 9600 bps:
}


void loop() {
  // put your main code here, to run repeatedly:
  Serial.print("----------------------------------------------------------------------\n");
  Serial.print("----------------------------------------------------------------------\n");
  
  // --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
  int n = 10; // This variable will contain the number of samples that the system under study will have.
  int m = 1; // This variable will contain the number of features (independent variables) that the input matrix is expected to have.
  int p = 1; // This variable will contain the number of outputs that the output matrix is expected to have.
  float b_ideal[2]; // This variable will be used to contain the ideal coefficient values that the model to be trained should give.
  b_ideal[0] = 10; // This is the true b_0 coefficient value with which the database was genereated.
  b_ideal[1] = 0.8; // This is the true b_1 coefficient value with which the database was genereated.

  
  // ---------------------- IMPORT DATA TO USE --------------------- //
  // We declare the input and output variables with the exact same input and output data as in the CenyML linear regression equation system that has 10 samples.
  float X[n] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100}; // This variable contains the input data of the system under study.
  float Y[n] = {18, 26, 34, 42, 50, 58, 66, 74, 82, 90}; // This variable ocntains the expected/real output data of the system under study.
  
  
  // ------------------------- DATA MODELING ----------------------- //
  Serial.print("\n\nInitializing CenyML simple linear regression algorithm ...\n");
  float startingTime = millis(); // We obtain the reference time to count the elapsed time to apply the simple linear regression with the input data (X).
  // Allocate the memory required for the variable "b", which will contain the identified best fitting coefficient values that will result from the simple linear regression algorithm.
  float *b = (float *) calloc(m+1, sizeof(float));
  // We apply the simple linear regression algorithm with respect to the input matrix "X" and the result is stored in the memory location of the pointer "b".
  getSimpleLinearRegression(X, Y, n, m, p, b); // NOTE: Remember that this functions stores the resulting coefficients in the pointer variable "b".
  float elapsedTime = millis() - startingTime; // We obtain the elapsed time to apply the simple linear regression with the input data (X).
  Serial.print("CenyML simple linear regression algorithm elapsed ");
  Serial.print(elapsedTime);
  Serial.print(" milliseconds.\n\n");


  // ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
  // We predict the input values (X) with the machine learning model that was obtained.
  Serial.print("Initializing CenyML predictions with the model that was obtained ...\n");
  startingTime = millis(); // We obtain the reference time to count the elapsed time to apply the prediction with the model that was obtained.
  // Allocate the memory required for the variable "Y_hat", which will contain the predicted output data of the system under study.
  float *Y_hat = (float *) malloc(n*sizeof(float));
  // We obtain the predicted values with the machine learning model that was obtained.
  predictSimpleLinearRegression(X, b, n, m, p, Y_hat); // NOTE: Remember that this functions stores the resulting coefficients in the pointer variable "Y_hat".
  elapsedTime = millis() - startingTime; // We obtain the elapsed time to obtain the prediction wit hthe model that was obtained.
  Serial.print("Making a prediction with the obtained model = ");
  Serial.print(Y_hat[2]);
  Serial.print("\n");
  Serial.print("CenyML predictions with the model that was obtained elapsed ");
  Serial.print(elapsedTime);
  Serial.print(" milliseconds.\n\n");

  
  // ------------------- VALIDATION OF THE MODEL ------------------- //
  // We validate the getSimpleLinearRegression method.
  Serial.print("Initializing coefficients validation of the CenyML getSimpleLinearRegression method ...\n");
  startingTime = millis(); // We obtain the reference time to count the elapsed time to validate the getSimpleLinearRegression method.
  float differentiation; // Variable used to store the error obtained for a certain value.
  float epsilon = 1.0E-8; // Variable used to store the max error value permitted during validation process.
  char isMatch = 1; // Variable used as a flag to indicate if the current comparation of values stands for a match. Note that the value of 1 = is a match and 0 = is not a match.
  // We check that all the differentiations do not surpass the error indicated through the variable "epsilon".
  for (int currentRow=0; currentRow<m+1; currentRow++) {
    differentiation = fabs(b[currentRow] - b_ideal[currentRow]);
    if (differentiation > epsilon) { // if the error surpassed the value permitted, then terminate validation process and emit message to indicate a non match.
      isMatch = 0;
      Serial.print("Validation process DID NOT MATCH! and a difference of ");
      Serial.print(differentiation);
      Serial.print(" was obtained.\n");
      break;
    }
  }
  if (isMatch == 1) { // If the flag "isMatch" indicates a true/high value, then emit message to indicate that the validation process matched.
    Serial.print("Validation process MATCHED!\n");
  }
  elapsedTime = millis() - startingTime; // We obtain the elapsed time to validate the getSimpleLinearRegression method.
  Serial.print("The coefficients validation of the CenyML getSimpleLinearRegression method elapsed ");
  Serial.print(elapsedTime);
  Serial.print(" milliseconds.\n\n");
  Serial.print("The program has been successfully completed!\n");

  // Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
  free(b);
  free(Y_hat);
  Serial.print("----------------------------------------------------------------------\n");
  Serial.print("----------------------------------------------------------------------\n");
  
  
}
