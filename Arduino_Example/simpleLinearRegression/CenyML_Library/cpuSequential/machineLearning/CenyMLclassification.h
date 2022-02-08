/*
   Copyright 2021 Cesar Miranda Meza (alias: Mortrack)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#ifndef CENYMLCLASSIFICATION_H
#define CENYMLCLASSIFICATION_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void getLinearLogisticClassification(float *, float *, int, int, int, float, char, float *);
void predictLinearLogisticClassification(float *, float, float *, int, int, int, float *);
void getSimpleLinearMachineClassification(float *, float *, int, int, int, char, float *);
void predictSimpleLinearMachineClassification(float *, float *, int, int, int, float *);
void getKernelMachineClassification(float *, float *, int, int, int, int, float, char[], char, char, char, float *);
static void trainLinearKernel(float *, float *, int, int, int, char, float *);
static void trainPolynomialKernel(float *, float *, int, int, int, int, char, char, float *);
static void trainLogisticKernel(float *, float *, int, int, int, char, float *);
static void trainGaussianKernel(float *, float *, int, int, int, float, char, char, float *);
void predictKernelMachineClassification(float *, int, char[], char, char, float *, int, int, int, float *);




/**
* The "getLinearLogisticClassification()" function is used to apply
* the machine learning algorithm called linear logistic
* classification as formulated in the master thesis of Cesar
* Miranda Meza called "Machine learning to support applications with
* embedded systems and parallel computing". Within this process, the
* best fitting equation with the form of "y_hat = 1 / (1+e^{-(b_{0}
* + b_{1}x_{1} + b_{2}x_{2} + ... + b_{m}x_{m})})" will be
* identified with respect to the sampled data given through the
* argument pointer variables "X" and "Y". At the end of this
* algorithm, the identified coefficient values will be stored in the
* argument pointer variable "b". As a result, when inserting the
* coefficient values into this model, whenever its output is greater
* than the defined threshold (0 < threshold < 1), it should be
* interpreted as the model predicting that the current input values
* represent group 1 or the binary number "1". Conversely, if the
* model produces a value less than the defined threshold, it should
* be interpreted as the model predicting that the current input
* values represent group 2 or the binary number "0".
*
* NOTE: The algorithm section that applied the matrix inverse using
* the Gauss-Jordan method was inspired in the following source:
* "CodeSansar. Matrix Inverse Using Gauss Jordan Method C Program.
* November 16, 2021 (Recovery date), de CodeSansar Sitio web:
* https://bit.ly/3CowwSy".
*
* 
* @param float *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'float' MEMORY
*					 SPACES.
*
* @param float *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A SIZE OF
*					 "n" TIMES "p=1" 'float' MEMORY SPACES.
*					 Finally, make sure that the output values are
*					 either "1" or "0".
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param float Y_epsilon -  This argument will contain the user defined
*							 epsilon value that will be used to temporarly
*							 store the sum of any "0" value with it and
*							 substract it to any "1" value of the output
*							 matrix ("Y"). This process is a strictly
*							 needed mathematical operation in the calculus
*							 of the desired error metric. If not followed,
*							 a mathematical error will be obtained due to
*							 the calculation of ln(0). IMPORTANT NOTE: The
*							 results will be temporarly stored so that the
*							 values of the output matrixes are not
*							 modified.
*
* @param char isVariableOptimizer = This argument variable is not
*									having any effect on this function
*									at the moment, as its functionality
*									has not been developed. However, it
*									is recommended to initialize it with
*									an integer value of zero so that it
*									does not surprise the user when it
*									gets developed in the near future.
*
* @param float *b - This argument will contain the pointer to a
*					 memory allocated variable in which we will store
*					 the identified best fitting coefficient values
*					 for the desired machine learning algorithm. These
*					 coefficients will each be stored in the same
*					 column but under different rows where the first
*					 coefficient (b_0) will be stored in the row with
*					 index 0 and the last coefficient (b_m) will be
*					 stored in the row with index "m". IT IS
*					 INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*					 AND INITIALIZED WITH ZEROS BEFORE CALLING THIS
*					 FUNCTION WITH A VARIABLE SIZE OF "m+1" TIMES "p=1"
*					 'float' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getLinearLogisticClassification(float *X, float *Y, int n, int m, int p, float Y_epsilon, char isVariableOptimizer, float *b) {
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m < 1) {
		Serial.print("\nERROR: The machine learning features (independent variables) must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the samples are less than the number of machine learning features, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (n < m) {
		Serial.print("\nERROR: The number of samples provided must be equal or higher than the number of machine learning features (independent variables) for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		Serial.print("\nERROR: With respect to the system under study, there must only be only one output for this particular algorithm.\n");
		exit(1);
	}
	// If the output values of the system under study has a value that is not within the restrictions requested for this algorithm, then emit an error message and terminate the program.
	for (int currentRow=0; currentRow<n; currentRow++) {
		if ((Y[currentRow]!=0) && (Y[currentRow]!=1)) {
			Serial.print("\nERROR: The output data had a value that is different than \"0\" or \"1\". Please assign the proper output values for this algorithm, considering the possible outputs of 0 and 1.\n");
			exit(1);
		}
	}
	
	// --------------- PREPROCESSING OF THE OUTPUT DATA --------------- //
	// Store the data that must be contained in the output matrix "Y_tilde".
	float *Y_tilde = (float *) malloc(n*p*sizeof(float)); // This variable will contain the output data of the system under study ("Y") as required by the training of this algorithm.
	for (int currentRow=0; currentRow<n; currentRow++) {
		if (Y[currentRow] == 1) { // Apply the epsilon differentiation that was requested in the arguments of this function.
			Y_tilde[currentRow] = Y[currentRow] - Y_epsilon;
		} else if (Y[currentRow] == 0) {
			Y_tilde[currentRow] = Y[currentRow] + Y_epsilon;
		}
		Y_tilde[currentRow] = log(Y_tilde[currentRow]/(1-Y_tilde[currentRow])); // Store the currently modified output value.
	}
	
	// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
	// Store the data that must be contained in the input matrix "X_tilde". In addition, we obtain the transpose of "X_tilde".
	int currentRowTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	float *X_tilde = (float *) malloc(n*mPlusOne*sizeof(float)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
	float *TransposeOf_X_tilde = (float *) malloc(mPlusOne*n*sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
	for (int currentRow=0; currentRow<n; currentRow++) {
		currentRow2 = 0; // We reset the counters used in the following for-loop.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowTimesM = currentRow*m;
		X_tilde[currentRowTimesMplusOne] = 1;
		TransposeOf_X_tilde[currentColumn2] = X_tilde[currentRowTimesMplusOne];
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			X_tilde[currentRowAndColumn] = X[currentColumn-1 + currentRowTimesM];
			currentRow2++;
			TransposeOf_X_tilde[currentColumn2 + currentRow2*n] = X_tilde[currentRowAndColumn];
		}
		currentColumn2++;
	}
	
	// -------------------- SOLUTION OF THE MODEL -------------------- //
	// In order to start obtaining the coefficients, we multiply the matrix "TransposeOf_X_tilde" with the matrix "X_tilde" and store the result in the matrix "matMul1".
	int currentColumnTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentColumnTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	float *matMul1 = (float *) calloc(mPlusOne*mPlusOne, sizeof(float)); // We allocate, and initialize with zeros, the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "X_tilde" and its transpose.
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowTimesN = currentRow*n;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			currentColumnTimesN = currentColumn*n;
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
				// Here we want to multiply "TransposeOf_X_tilde" with the matrix "X_tilde", but we will use "TransposeOf_X_tilde" for such multiplication since they contain the same data, for performance purposes.
				matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesN] * TransposeOf_X_tilde[currentMultipliedElements + currentColumnTimesN];
			}
	    }
	}
	
	// In order to continue obtaining the coefficients, we innitialize the data of a unitary matrix with the same dimensions of the matrix "matMul1".
	// NOTE: Because getting the data of the transpose of "X_tilde"
	//		 directly from that same variable ("X_tilde"), will
	//		 increase performance in further steps, we will store the
	//		 matrix inverse of "matMul1" in the variable
	//		 "TransposeOf_X_tilde", in order to maximize computational
	//		 resources and further increase performance by not having to
	//		 allocate more memory in the computer system.
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // We set all values to zero.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			TransposeOf_X_tilde[currentColumn + currentRowTimesMplusOne] = 0;
	    }
	}
	for (int currentUnitaryValue=0; currentUnitaryValue<mPlusOne; currentUnitaryValue++) { // We set the corresponding 1's values to make the corresponding unitary matrix.
		TransposeOf_X_tilde[currentUnitaryValue + currentUnitaryValue*mPlusOne] = 1;
	}
	
	// In order to continue obtaining the coefficients, we calculate the matrix inverse of "matMul1" with the Gauss-Jordan approach and store its result in "TransposeOf_X_tilde".
	int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	float ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
	for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) { // We apply the differentiations applied to each row according to the approach used.
		currentColumnTimesMplusOne = currentColumn*mPlusOne;
		currentRowAndColumn2 = currentColumn + currentColumnTimesMplusOne;
		for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
			if (currentRow != currentColumn) {
				currentRowTimesMplusOne = currentRow*mPlusOne;
				ratioModifier = matMul1[currentColumn + currentRowTimesMplusOne]/matMul1[currentRowAndColumn2];
				for (int currentModifiedElements=0; currentModifiedElements<mPlusOne; currentModifiedElements++) { // We apply the current process to the principal matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesMplusOne;
					matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] - ratioModifier * matMul1[currentModifiedElements + currentColumnTimesMplusOne];
				}
				for (int currentModifiedElements=0; currentModifiedElements<mPlusOne; currentModifiedElements++) { // We apply the current process to the result matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesMplusOne;
					TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] - ratioModifier * TransposeOf_X_tilde[currentModifiedElements + currentColumnTimesMplusOne];
				}
			}
		}
    }
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // We apply the last step of the approach used in order to obtain the diagonal of 1's in the principal matrix.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowAndColumn2 = currentRow + currentRowTimesMplusOne;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] / matMul1[currentRowAndColumn2];
		}
    }
    
	// In order to continue obtaining the coefficients, we multiply the inverse matrix that was obtained by the transpose of the matrix "X_tilde".
	// NOTE: Remember that we will get the data of the transpose of
	//		 "X_tilde" directly from that same variable
	//		 ("X_tilde") due to performance reasons and; that the
	//		 inverse matrix that was obtained is stored in
	//		 "TransposeOf_X_tilde".
	float *matMul2 = (float *) calloc(mPlusOne*n, sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "X_tilde".
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
		currentRowTimesN = currentRow*n;
		currentRowTimesMplusOne = currentRow*mPlusOne;
		for (int currentColumn=0; currentColumn<n; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesN;
			currentColumnTimesMplusOne = currentColumn*mPlusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<mPlusOne; currentMultipliedElements++) {
				matMul2[currentRowAndColumn] = matMul2[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesMplusOne] * X_tilde[currentMultipliedElements + currentColumnTimesMplusOne];
			}
	    }
	}
	
	// In order to conclude obtaining the coefficients ("b"), we multiply the previously resulting matrix ("matMul2") by the output matrix "Y_tilde".
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
		currentRowTimesN = currentRow*n;
		for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
			b[currentRow] = b[currentRow] + matMul2[currentMultipliedElements + currentRowTimesN] * Y_tilde[currentMultipliedElements];
		}
	}
	
	// Free the Heap memory used for the locally allocated variables since they will no longer be used.
	free(Y_tilde);
	free(X_tilde);
	free(TransposeOf_X_tilde);
	free(matMul1);
	free(matMul2);
	return;
}


/**
* The "predictLinearLogisticClassification()" function is used to
* make the predictions of the requested input values (X) by applying
* the linear logistic classification model with the specified
* coefficient values (b). The predicted values will be stored in the
* argument pointer variable "Y_hat".
* 
* @param float *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning predictions will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'float' MEMORY
*					 SPACES.
*
* @param float threshold - This argument will represent the value
*							that the implementer desires for the
*							threshold to be taken into account
*							during the predictions made by the
*							machine learning model that was trained.
*							Moreover, keep in mind the restriction
*							"0 < threshold < 1", which must be
*							complied.
*
* @param float *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE IS ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A VARIABLE
*					 SIZE OF "m+1" TIMES "p=1" 'float' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param float *Y_hat - This argument will contain the pointer to a
*					 	 memory allocated output matrix, representing
*					 	 the predicted data of the system under study.
*						 THIS VARIABLE SHOULD BE ALLOCATED BEFORE
*						 CALLING THIS FUNCTION WITH A SIZE OF "n"
*						 TIMES "p=1" 'float' MEMORY SPACES. The
*						 results will be stored in the same order as
*						 the input data given such that the first
*						 sample will be stored in the row with index
*						 "0" and the last sample in the row with
*						 index "n". Finally, the possible output
*						 values are either "1" or "0".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 24, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void predictLinearLogisticClassification(float *X, float threshold, float *b, int n, int m, int p, float *Y_hat) {
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m < 1) {
		Serial.print("\nERROR: The machine learning features (independent variables) must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		Serial.print("\nERROR: With respect to the system under study, there must only be only one output for this particular algorithm.\n");
		exit(1);
	}
	// If the specified threshold does not complied with the required restriction, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((threshold<=0) || (threshold>=1)) {
		Serial.print("\nERROR: The specified threshold does not meet the restriction: 0 < threshold < 1.\n");
		exit(1);
	}
	
	
	// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (b).
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow=0; currentRow<n; currentRow++) {
		// We calculate the current probability given by the model.
		Y_hat[currentRow] = b[0];
		currentRowTimesM = currentRow*m;
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			Y_hat[currentRow] = Y_hat[currentRow] + b[currentColumn]*X[currentColumn-1 + currentRowTimesM];
		}
		Y_hat[currentRow] = 1 / (1 + exp(-Y_hat[currentRow]));
		
		// We determine the prediction made by the model based on whether its given current probability is greater than the defined threshold or not.
		if (Y_hat[currentRow] > threshold) {
			Y_hat[currentRow] = 1;
		} else {
			Y_hat[currentRow] = 0;
		}
	}
	
	return;
}


/**
* The "getSimpleLinearMachineClassification()" function is used to
* apply the machine learning algorithm called simple linear machine
* classification as formulated in the master thesis of Cesar Miranda
* Meza called "Machine learning to support applications with
* embedded systems and parallel computing". Within this process, the
* best fitting equation with the form of "y_hat = omega.x^T + b_0"
* ("omega" and "b_0" are the coefficients to be identified, "omega"
* is a vector containing the coefficients of all the machine
* learning features, "b_0" is the bias coefficient of the model, "T"
* stands for transpose and "x" is a vector containing all the
* machine learning features) will be identified with respect to the
* sampled data given through the argument pointer variables "X" and
* "Y". At the end of this algorithm, the identified coefficient
* values will be stored in the argument pointer variable "b". As a
* result, when inserting the coefficient values into this model,
* whenever its output is greater than the value of "0", it should be
* interpreted as the model predicting that the current input values
* represent group 1 or the numeric value of "+1". Conversely, if the
* model produces a value less than "0", it should be interpreted as
* the model predicting that the current input values represent group
* 2 or the numeric value of "-1".
*
* NOTE: The algorithm section that applied the matrix inverse using
* the Gauss-Jordan method was inspired in the following source:
* "CodeSansar. Matrix Inverse Using Gauss Jordan Method C Program.
* November 16, 2021 (Recovery date), de CodeSansar Sitio web:
* https://bit.ly/3CowwSy".
*
* 
* @param float *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'float' MEMORY
*					 SPACES.
*
* @param float *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A SIZE OF
*					 "n" TIMES "p=1" 'float' MEMORY SPACES.
*					 Finally, make sure that the output values are
*					 either "+1" or "-1".
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param char isVariableOptimizer = This argument variable is not
*									having any effect on this function
*									at the moment, as its functionality
*									has not been developed. However, it
*									is recommended to initialize it with
*									an integer value of zero so that it
*									does not surprise the user when it
*									gets developed in the near future.
*
* @param float *b - This argument will contain the pointer to a
*					 memory allocated variable in which we will store
*					 the identified best fitting coefficient values
*					 for the desired machine learning algorithm. These
*					 coefficients will each be stored in the same
*					 column but under different rows where the first
*					 coefficient (b_0) will be stored in the row with
*					 index 0 and the last coefficient (b_m or, in other
*					 words, omega_m) will be stored in the row with
*					 index "m". IT IS INDISPENSABLE THAT THIS VARIABLE
*					 IS ALLOCATED AND INITIALIZED WITH ZEROS BEFORE
*					 CALLING THIS FUNCTION WITH A VARIABLE SIZE OF "m+1"
*					 TIMES "p=1" 'float' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 24, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getSimpleLinearMachineClassification(float *X, float *Y, int n, int m, int p, char isVariableOptimizer, float *b) {
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m < 1) {
		Serial.print("\nERROR: The machine learning features (independent variables) must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the samples are less than the number of machine learning features, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (n < m) {
		Serial.print("\nERROR: The number of samples provided must be equal or higher than the number of machine learning features (independent variables) for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		Serial.print("\nERROR: With respect to the system under study, there must only be only one output for this particular algorithm.\n");
		exit(1);
	}
	// If the output values of the system under study has a value that is not within the restrictions requested for this algorithm, then emit an error message and terminate the program.
	for (int currentRow=0; currentRow<n; currentRow++) {
		if ((Y[currentRow]!=-1) && (Y[currentRow]!=1)) {
			Serial.print("\nERROR: The output data had a value that is different than \"-1\" or \"+1\". Please assign the proper output values for this algorithm, considering the possible outputs of -1 and +1.\n");
			exit(1);
		}
	}
	
	// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
	// Store the data that must be contained in the input matrix "X_tilde". In addition, we obtain the transpose of "X_tilde".
	int currentRowTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	float *X_tilde = (float *) malloc(n*mPlusOne*sizeof(float)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
	float *TransposeOf_X_tilde = (float *) malloc(mPlusOne*n*sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
	for (int currentRow=0; currentRow<n; currentRow++) {
		currentRow2 = 0; // We reset the counters used in the following for-loop.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowTimesM = currentRow*m;
		X_tilde[currentRowTimesMplusOne] = 1;
		TransposeOf_X_tilde[currentColumn2] = X_tilde[currentRowTimesMplusOne];
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			X_tilde[currentRowAndColumn] = X[currentColumn-1 + currentRowTimesM];
			currentRow2++;
			TransposeOf_X_tilde[currentColumn2 + currentRow2*n] = X_tilde[currentRowAndColumn];
		}
		currentColumn2++;
	}
	
	// -------------------- SOLUTION OF THE MODEL -------------------- //
	// In order to start obtaining the coefficients, we multiply the matrix "TransposeOf_X_tilde" with the matrix "X_tilde" and store the result in the matrix "matMul1".
	int currentColumnTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentColumnTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	float *matMul1 = (float *) calloc(mPlusOne*mPlusOne, sizeof(float)); // We allocate, and initialize with zeros, the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "X_tilde" and its transpose.
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowTimesN = currentRow*n;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			currentColumnTimesN = currentColumn*n;
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
				// Here we want to multiply "TransposeOf_X_tilde" with the matrix "X_tilde", but we will use "TransposeOf_X_tilde" for such multiplication since they contain the same data, for performance purposes.
				matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesN] * TransposeOf_X_tilde[currentMultipliedElements + currentColumnTimesN];
			}
	    }
	}
	
	// In order to continue obtaining the coefficients, we innitialize the data of a unitary matrix with the same dimensions of the matrix "matMul1".
	// NOTE: Because getting the data of the transpose of "X_tilde"
	//		 directly from that same variable ("X_tilde"), will
	//		 increase performance in further steps, we will store the
	//		 matrix inverse of "matMul1" in the variable
	//		 "TransposeOf_X_tilde", in order to maximize computational
	//		 resources and further increase performance by not having to
	//		 allocate more memory in the computer system.
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // We set all values to zero.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			TransposeOf_X_tilde[currentColumn + currentRowTimesMplusOne] = 0;
	    }
	}
	for (int currentUnitaryValue=0; currentUnitaryValue<mPlusOne; currentUnitaryValue++) { // We set the corresponding 1's values to make the corresponding unitary matrix.
		TransposeOf_X_tilde[currentUnitaryValue + currentUnitaryValue*mPlusOne] = 1;
	}
	
	// In order to continue obtaining the coefficients, we calculate the matrix inverse of "matMul1" with the Gauss-Jordan approach and store its result in "TransposeOf_X_tilde".
	int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	float ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
	for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) { // We apply the differentiations applied to each row according to the approach used.
		currentColumnTimesMplusOne = currentColumn*mPlusOne;
		currentRowAndColumn2 = currentColumn + currentColumnTimesMplusOne;
		for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
			if (currentRow != currentColumn) {
				currentRowTimesMplusOne = currentRow*mPlusOne;
				ratioModifier = matMul1[currentColumn + currentRowTimesMplusOne]/matMul1[currentRowAndColumn2];
				for (int currentModifiedElements=0; currentModifiedElements<mPlusOne; currentModifiedElements++) { // We apply the current process to the principal matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesMplusOne;
					matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] - ratioModifier * matMul1[currentModifiedElements + currentColumnTimesMplusOne];
				}
				for (int currentModifiedElements=0; currentModifiedElements<mPlusOne; currentModifiedElements++) { // We apply the current process to the result matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesMplusOne;
					TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] - ratioModifier * TransposeOf_X_tilde[currentModifiedElements + currentColumnTimesMplusOne];
				}
			}
		}
    }
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // We apply the last step of the approach used in order to obtain the diagonal of 1's in the principal matrix.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowAndColumn2 = currentRow + currentRowTimesMplusOne;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] / matMul1[currentRowAndColumn2];
		}
    }
    
	// In order to continue obtaining the coefficients, we multiply the inverse matrix that was obtained by the transpose of the matrix "X_tilde".
	// NOTE: Remember that we will get the data of the transpose of
	//		 "X_tilde" directly from that same variable
	//		 ("X_tilde") due to performance reasons and; that the
	//		 inverse matrix that was obtained is stored in
	//		 "TransposeOf_X_tilde".
	float *matMul2 = (float *) calloc(mPlusOne*n, sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "X_tilde".
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
		currentRowTimesN = currentRow*n;
		currentRowTimesMplusOne = currentRow*mPlusOne;
		for (int currentColumn=0; currentColumn<n; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesN;
			currentColumnTimesMplusOne = currentColumn*mPlusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<mPlusOne; currentMultipliedElements++) {
				matMul2[currentRowAndColumn] = matMul2[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesMplusOne] * X_tilde[currentMultipliedElements + currentColumnTimesMplusOne];
			}
	    }
	}
	
	// In order to conclude obtaining the coefficients ("b"), we multiply the previously resulting matrix ("matMul2") by the output matrix "Y".
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
		currentRowTimesN = currentRow*n;
		for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
			b[currentRow] = b[currentRow] + matMul2[currentMultipliedElements + currentRowTimesN] * Y[currentMultipliedElements];
		}
	}
	
	// Free the Heap memory used for the locally allocated variables since they will no longer be used.
	free(X_tilde);
	free(TransposeOf_X_tilde);
	free(matMul1);
	free(matMul2);
	return;
}


/**
* The "predictSimpleLinearMachineClassification()" function is used to
* make the predictions of the requested input values (X) by applying
* the simple linear machine classification model with the specified
* coefficient values (b). The predicted values will be stored in the
* argument pointer variable "Y_hat".
* 
* @param float *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning predictions will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'float' MEMORY
*					 SPACES.
*
* @param float *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE IS ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A VARIABLE
*					 SIZE OF "m+1" TIMES "p=1" 'float' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param float *Y_hat - This argument will contain the pointer to a
*					 	 memory allocated output matrix, representing
*					 	 the predicted data of the system under study.
*						 THIS VARIABLE SHOULD BE ALLOCATED BEFORE
*						 CALLING THIS FUNCTION WITH A SIZE OF "n"
*						 TIMES "p=1" 'float' MEMORY SPACES. The
*						 results will be stored in the same order as
*						 the input data given such that the first
*						 sample will be stored in the row with index
*						 "0" and the last sample in the row with
*						 index "n". Finally, the possible output
*						 values are either "+1" or "-1".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 24, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void predictSimpleLinearMachineClassification(float *X, float *b, int n, int m, int p, float *Y_hat) {
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m < 1) {
		Serial.print("\nERROR: The machine learning features (independent variables) must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		Serial.print("\nERROR: With respect to the system under study, there must only be only one output for this particular algorithm.\n");
		exit(1);
	}
	
	// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (b).
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow=0; currentRow<n; currentRow++) {
		Y_hat[currentRow] = b[0];
		currentRowTimesM = currentRow*m;
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			Y_hat[currentRow] = Y_hat[currentRow] + b[currentColumn]*X[currentColumn-1 + currentRowTimesM];
		}
		if (Y_hat[currentRow] > 0) {
			Y_hat[currentRow] = 1;
		} else {
			Y_hat[currentRow] = -1;
		}
	}
	
	return;
}


/**
* The "getKernelMachineClassification()" function is used to
* apply the machine learning algorithm called Kernel machine
* classification. Within this process, the best fitting equation
* with the form of "y_hat = alpha.K(x) + b_0" will be identified
* with respect to the sampled data given through the argument
* pointer variables "X" and "Y". In that equation, "alpha" and
* "b_0" are the coefficients to be identified, "K(x)" is the
* Kernel function which is basically a transformation function
* of x, "alpha" is the coefficient of the Kernel, "b_0" is the
* bias coefficient of the model, "x" is a vector containing all
* the machine learning features and the coefficients within
* "K(x)" will be representend as "Beta". The way this algorithm
* is going to be solved will be as formulated in the master
* thesis of Cesar Miranda Meza called "Machine learning to
* support applications with embedded systems and parallel
* computing". Moreover, when inserting all the identified
* coefficient values into this model, whenever its output is
* greater than the value of "0", it should be interpreted as the
* model predicting that the current input values represent group
* 1 or the numeric value of "+1". Conversely, if the model
* produces a value less than "0", it should be interpreted as
* the model predicting that the current input values represent
* group 2 or the numeric value of "-1".
*
* THIS IS IMPORTANT TO KNOW --> Finally, have in mind that not
* all the argument variables of this function will be used and
* the ones that do will strictly depend on what Kernel you
* tell this function to use. Therefore, make sure to read the
* following descriptions made for all the possible argument
* variables to be taken into consideration when using this
* function. There, it will be explicitly specified whether such
* argument variable will be used and under what cirumstances
* and what to do when it will not be used.
* 
* @param float *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'float' MEMORY
*					 SPACES.
*
* @param float *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A SIZE OF
*					 "n" TIMES "p=1" 'float' MEMORY SPACES.
*					 Finally, make sure that the output values are
*					 either "+1" or "-1".
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param int N - This argument will only used by this function whenever
*				 the "polynomial" Kernel is selected with the argument
*				 variable "Kernel". For the case in which a different
*				 Kernel is chosen, then it is suggested to assign a
*				 value of "(int) 0" to "N". Conversely, if the
*				 "polynomial" Kernel is selected, then this argument
*				 variable will represent the desired order of degree
*				 for the machine learning model to be trained.
*
* @param float zeroEpsilon - This argument will only be used whenever the
*							  "gaussian" Kernel is selected. For the case
*							  in which a different Kernel is chosen, then
*							  it is suggested to assign a value of
*							  "(float) 0" to "zeroEpsilon". Conversely,
*							  if the "gaussian" Kernel is selected, then
*							  this argument variable will contain the user
*							  defined epsilon value that will be used to
*							  temporarly store any "0", that is contained
*							  in the output matrix "Y", with the added
*							  value of "zeroEpsilon" but only for the
*							  gaussian regression training process. The
*							  process in which the "zeroEpsilon" is part
*							  of, is a strictly needed mathematical
*							  operation in the calculus of the output
*							  transformation "Y_tilde = ln(Y)" of the
*							  mentioned regression algorithm. The
*							  restriction that has to be complied is
*							  "0 < zeroEpsilon < 1" and its value will
*							  change the bias of the gaussian Kernel.
*							  Ussually a good value is "zeroEpsilon = 0.1"
*							  but other values can be tried in order to
*							  better fit the model.
*							  IMPORTANT NOTE: The results will be
*							  temporarly stored so that the values of the
*							  output matrix "Y" is not modified.
*
* @param char Kernel[] - This argument will contain the string or array
*						 of characters that will specify the Kernel
*						 type requested by the implementer. Its
*						 possible values are the following:
*						 1) "linear" = applies the linear kernel.
*						 2) "polynomial" = applies the polynomial
*										   kernel.
*						 3) "logistic" = applies the logistic kernel.
*						 4) "gaussian" = applies the gaussian kernel.
*
* @param char isInteractionTerms = This argument variable is only used
*								   by this function whenever the
*								   "polynomial" Kernel is selected with
*								   the argument variable "Kernel". For
*								   the case in which a different Kernel
*								   is chosen, then assign a value of
*								   (int) 0 to this argument variable.
*								   Conversely, if you select the
*								   "polynomial" Kernel, then this
*								   argument variable will work as a flag
*								   to indicate whether the interaction
*								   terms are desired in the resulting
*								   model to be generated or not.
*						  		   Moreover, the possible valid values
*								   for this argument variable are:
*						  		   1) (int) 1 = The resulting model will
*												include the interaction
*												terms that are possible.
*												NOTE: This functionality,
*												is yet to be developed.
*						  		   2) (int) 0 = The resulting model will
*												not include interaction
*												terms.
*
* @param char isForceGaussianCurve = This argument variable will only be
*							used if the "gaussian" Kernel is chosen. For
*							the case in which a different Kernel is
*							selected, then it is suggested to define a
*						    value of "(int) 0" to this argument variable.
*							Conversely, if the "gaussian" Kernel is
*							chosen, then "isForceGaussianCurve" will
*							determine the types of coefficients to be
*							obtained during the training process. If
*							"isForceGaussianCurve" = 0, then the
*							coefficients to be obtained will be
*							directly from a model that has an
*							exponential with the multiple polynomial
*							equation without its interaction terms from
*							the following mathematical form "y_hat =
*							exp( b_tilde_{0} + b_tilde_{1}*x_{1} +
*							b_tilde_{2}*x_{1}^2 + b_tilde_{3}*x_{2} +
* 							b_tilde_{4}*x_{2}^2 + ... +
*							b_tilde_{2*m-1}*x_{m} +
* 							b_tilde_{2*m}*x_{m}^2 )".
*							If "isForceGaussianCurve" = 1, then the
*							coefficients to be obtained will be in the
*							from of an approximation from the previous
*							equation, such that each machine learning
*							feature will have a particular "mean" and
*							"variance" coefficient values. These will be
*							used to govern the following model instead
*							"y_hat =
*							exp(-( ((x_{1}-mean_{1})^2)/2*variance_{1} +
* 							((x_{2}-mean_{2})^2)/2*variance_{2} + ... +
* 							((x_{m}-mean_{m})^2)/2*variance_{m} ))".
*							On the other hand, have in consideration
*							that because the solution of the multiple
*							polynomial equation can sometimes give a non
*							perfect square binomial, when
*							"isForceGaussianCurve" = 0, you may not get
*							a true gaussian form under such circumstance.
*							Therefore, by setting "isForceGaussianCurve"
*							= 1, you will force the solution to always
*							have a gaussian form but this may increase
*							the total obtained error as a consequence.
*
* @param char isVariableOptimizer = This argument variable is not
*									having any effect on this function
*									at the moment, as its functionality
*									has not been developed. However, it
*									is recommended to initialize it with
*									an integer value of zero so that it
*									does not surprise the user when it
*									gets developed in the near future.
*
* @param float *coefficients - This argument will contain the pointer
*					 to a memory allocated variable in which we will
*					 store the identified best fitting coefficient values
*					 for the desired machine learning algorithm. These
*					 coefficients will each be stored in a different
*					 manner, depending on the Kernel that was chosen
*					 for the training process. The details of this will
*					 be described in the particular static function
*					 that corresponds to the chosen Kernel. Because of
*					 this, the following will list the static function
*					 that will be called for each Kernel that may be
*					 chosen so that the implementer reads their
*					 commented documentation to learn the details of
*					 how much memory to allocate in the argument pointer
*					 variable "coefficients" and to know how the data
*					 will be stored in it.
*					 1) Kernel="linear"  -->  trainLinearKernel()
*					 2) Kernel="polynomial"  -->  trainPolynomialKernel()
*					 3) Kernel="logistic"  -->  trainLogisticKernel()
*					 4) Kernel="gaussian"  -->  trainGaussianKernel()
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"coefficients".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 27, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getKernelMachineClassification(float *X, float *Y, int n, int m, int p, int N, float zeroEpsilon, char Kernel[], char isInteractionTerms, char isForceGaussianCurve, char isVariableOptimizer, float *coefficients) {
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m < 1) {
		Serial.print("\nERROR: The machine learning features (independent variables) must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the samples are less than the number of machine learning features, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (n < m) {
		Serial.print("\nERROR: The number of samples provided must be equal or higher than the number of machine learning features (independent variables) for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		Serial.print("\nERROR: With respect to the system under study, there must only be only one output for this particular algorithm.\n");
		exit(1);
	}
	// If the output values of the system under study has a value that is not within the restrictions requested for this algorithm, then emit an error message and terminate the program.
	for (int currentRow=0; currentRow<n; currentRow++) {
		if ((Y[currentRow]!=-1) && (Y[currentRow]!=1)) {
			Serial.print("\nERROR: The output data had a value that is different than \"-1\" or \"+1\". Please assign the proper output values for this algorithm, considering the possible outputs of -1 and +1.\n");
			exit(1);
		}
	}
	
	// Apply the Kernel that was requested by the implementer of this function.
	if (strcmp(Kernel, "linear") == 0) {
		trainLinearKernel(X, Y, n, m, p, isVariableOptimizer, coefficients); // Train the Kernel machine classifier with a linear kernel.
	} else if (strcmp(Kernel, "polynomial") == 0) {
		trainPolynomialKernel(X, Y, n, m, p, N, isInteractionTerms, isVariableOptimizer, coefficients); // Train the Kernel machine classifier with a polynomial kernel.
	} else if (strcmp(Kernel, "logistic") == 0) {
		trainLogisticKernel(X, Y, n, m, p, isVariableOptimizer, coefficients); // Train the Kernel machine classifier with a logistic kernel.
	} else if (strcmp(Kernel, "gaussian") == 0) {
		trainGaussianKernel(X, Y, n, m, p, zeroEpsilon, isForceGaussianCurve, isVariableOptimizer, coefficients); // Train the Kernel machine classifier with a gaussian kernel.
	} else {
		Serial.print("\nERROR: The requested Kernel has not yet been implemented in the CenyML library. Please assign a valid one.\n");
		exit(1);
	}
	
	return;
}


/**
* The "trainLinearKernel()" is a static function that is used to
* apply the machine learning algorithm called Kernel machine
* classification with a linear Kernel. Within this process, the
* best fitting equation with the form of "y_hat = alpha.K(x) +
* b_0" will be identified with respect to the sampled data given
* through the argument pointer variables "X" and "Y". In that
* equation, "alpha" and "b_0" are the coefficients to be
* identified, "K(x)" is the Kernel function which is basically a
* transformation function of x, "alpha" is the coefficient of the
* Kernel, "b_0" is the bias coefficient of the model, "x" is a
* vector containing all the machine learning features and the
* coefficients within "K(x)" will be representend as "Beta". With
* this in mind and just like in the master thesis of Cesar
* Miranda Meza ("Machine learning to support applications with
* embedded systems and parallel computing"), the first step to
* train this model is to solve the kernel function. For this
* purpose and for this particular type of Kernel, a multiple
* linear regression will be applied with respect to the output
* of the system under study "Y". Both the characteristic
* equation of the multiple linear system and its best fitting
* coefficient values will together represent the Kernel function
* K(x). Now, for this particular Kernel machine classifier,
* because of being a linear system, the coefficient value of
* "alpha"=1 and "b_0"=0. As a result, when inserting all the
* coefficient values into this model, whenever its output is
* greater than the value of "0", it should be interpreted as the
* model predicting that the current input values represent group
* 1 or the numeric value of "+1". Conversely, if the model
* produces a value less than "0", it should be interpreted as the
* model predicting that the current input values represent group
* 2 or the numeric value of "-1".
*
* NOTE: The algorithm section that applied the matrix inverse using
* the Gauss-Jordan method was inspired in the following source:
* "CodeSansar. Matrix Inverse Using Gauss Jordan Method C Program.
* November 16, 2021 (Recovery date), de CodeSansar Sitio web:
* https://bit.ly/3CowwSy".
*
* 
* @param float *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'float' MEMORY
*					 SPACES.
*
* @param float *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A SIZE OF
*					 "n" TIMES "p=1" 'float' MEMORY SPACES.
*					 Finally, make sure that the output values are
*					 either "+1" or "-1".
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param char isVariableOptimizer = This argument variable is not
*									having any effect on this function
*									at the moment, as its functionality
*									has not been developed. However, it
*									is recommended to initialize it with
*									an integer value of zero so that it
*									does not surprise the user when it
*									gets developed in the near future.
*
* @param float *coefficients - This argument will contain the pointer
*					 to a memory allocated variable in which we will
*					 store the identified best fitting coefficient values
*					 for the desired machine learning algorithm. These
*					 coefficients will each be stored in the same
*					 column but under different rows where the first
*					 coefficient (Beta_0) will be stored in the row with
*					 index 0, the last Beta value (Beta_m) will be
*					 stored in the row with index "m", the bias
*					 coefficient value (b_0) will be stored in the row
*					 with index "m+1" and the last coefficient (alpha)
*					 will be stored in the row with index "m+2". NOTE:
*					 For more information on how the "Beta"
*					 coefficients will be stored, please see the
*					 description given for the argument pointer variable
*					 "b" of the function "getMultipleLinearRegression()"
*					 in the "CenyMLregression.c" file. IT IS
*					 INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED AND
*					 INITIALIZED WITH ZEROS BEFORE CALLING THIS
*					 FUNCTION WITH A VARIABLE SIZE OF "m+3" TIMES
*					 "p=1" 'float' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"coefficients".
*
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 26, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
static void trainLinearKernel(float *X, float *Y, int n, int m, int p, char isVariableOptimizer, float *coefficients) {
	// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
	// Store the data that must be contained in the input matrix "X_tilde". In addition, we obtain the transpose of "X_tilde".
	int currentRowTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	float *X_tilde = (float *) malloc(n*mPlusOne*sizeof(float)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
	float *TransposeOf_X_tilde = (float *) malloc(mPlusOne*n*sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
	for (int currentRow=0; currentRow<n; currentRow++) {
		currentRow2 = 0; // We reset the counters used in the following for-loop.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowTimesM = currentRow*m;
		X_tilde[currentRowTimesMplusOne] = 1;
		TransposeOf_X_tilde[currentColumn2] = X_tilde[currentRowTimesMplusOne];
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			X_tilde[currentRowAndColumn] = X[currentColumn-1 + currentRowTimesM];
			currentRow2++;
			TransposeOf_X_tilde[currentColumn2 + currentRow2*n] = X_tilde[currentRowAndColumn];
		}
		currentColumn2++;
	}
	
	// -------------------- SOLUTION OF THE MODEL -------------------- //
	// NOTE: To solve for the "Beta_0" + "Beta_1", ..., "Beta_m" coefficients, we will apply a multiple linear regression.
	// In order to start obtaining the coefficients, we multiply the matrix "TransposeOf_X_tilde" with the matrix "X_tilde" and store the result in the matrix "matMul1".
	int currentColumnTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentColumnTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	float *matMul1 = (float *) calloc(mPlusOne*mPlusOne, sizeof(float)); // We allocate, and initialize with zeros, the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "X_tilde" and its transpose.
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowTimesN = currentRow*n;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			currentColumnTimesN = currentColumn*n;
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
				// Here we want to multiply "TransposeOf_X_tilde" with the matrix "X_tilde", but we will use "TransposeOf_X_tilde" for such multiplication since they contain the same data, for performance purposes.
				matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesN] * TransposeOf_X_tilde[currentMultipliedElements + currentColumnTimesN];
			}
	    }
	}
	
	// In order to continue obtaining the coefficients, we innitialize the data of a unitary matrix with the same dimensions of the matrix "matMul1".
	// NOTE: Because getting the data of the transpose of "X_tilde"
	//		 directly from that same variable ("X_tilde"), will
	//		 increase performance in further steps, we will store the
	//		 matrix inverse of "matMul1" in the variable
	//		 "TransposeOf_X_tilde", in order to maximize computational
	//		 resources and further increase performance by not having to
	//		 allocate more memory in the computer system.
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // We set all values to zero.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			TransposeOf_X_tilde[currentColumn + currentRowTimesMplusOne] = 0;
	    }
	}
	for (int currentUnitaryValue=0; currentUnitaryValue<mPlusOne; currentUnitaryValue++) { // We set the corresponding 1's values to make the corresponding unitary matrix.
		TransposeOf_X_tilde[currentUnitaryValue + currentUnitaryValue*mPlusOne] = 1;
	}
	
	// In order to continue obtaining the coefficients, we calculate the matrix inverse of "matMul1" with the Gauss-Jordan approach and store its result in "TransposeOf_X_tilde".
	int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	float ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
	for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) { // We apply the differentiations applied to each row according to the approach used.
		currentColumnTimesMplusOne = currentColumn*mPlusOne;
		currentRowAndColumn2 = currentColumn + currentColumnTimesMplusOne;
		for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
			if (currentRow != currentColumn) {
				currentRowTimesMplusOne = currentRow*mPlusOne;
				ratioModifier = matMul1[currentColumn + currentRowTimesMplusOne]/matMul1[currentRowAndColumn2];
				for (int currentModifiedElements=0; currentModifiedElements<mPlusOne; currentModifiedElements++) { // We apply the current process to the principal matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesMplusOne;
					matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] - ratioModifier * matMul1[currentModifiedElements + currentColumnTimesMplusOne];
				}
				for (int currentModifiedElements=0; currentModifiedElements<mPlusOne; currentModifiedElements++) { // We apply the current process to the result matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesMplusOne;
					TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] - ratioModifier * TransposeOf_X_tilde[currentModifiedElements + currentColumnTimesMplusOne];
				}
			}
		}
    }
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // We apply the last step of the approach used in order to obtain the diagonal of 1's in the principal matrix.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowAndColumn2 = currentRow + currentRowTimesMplusOne;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] / matMul1[currentRowAndColumn2];
		}
    }
    
	// In order to continue obtaining the coefficients, we multiply the inverse matrix that was obtained by the transpose of the matrix "X_tilde".
	// NOTE: Remember that we will get the data of the transpose of
	//		 "X_tilde" directly from that same variable
	//		 ("X_tilde") due to performance reasons and; that the
	//		 inverse matrix that was obtained is stored in
	//		 "TransposeOf_X_tilde".
	float *matMul2 = (float *) calloc(mPlusOne*n, sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "X_tilde".
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
		currentRowTimesN = currentRow*n;
		currentRowTimesMplusOne = currentRow*mPlusOne;
		for (int currentColumn=0; currentColumn<n; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesN;
			currentColumnTimesMplusOne = currentColumn*mPlusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<mPlusOne; currentMultipliedElements++) {
				matMul2[currentRowAndColumn] = matMul2[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesMplusOne] * X_tilde[currentMultipliedElements + currentColumnTimesMplusOne];
			}
	    }
	}
	
	// In order to conclude obtaining the coefficients ("coefficients"), we multiply the previously resulting matrix ("matMul2") by the output matrix "Y".
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // In this for-loop, we will store the "Beta" coefficient values of the Kernel function "K(x)".
		currentRowTimesN = currentRow*n;
		for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
			coefficients[currentRow] = coefficients[currentRow] + matMul2[currentMultipliedElements + currentRowTimesN] * Y[currentMultipliedElements];
		}
	}
	coefficients[1+m] = 0; // We store the value of "b_0" from the main function to be solved, which is "y_hat = alpha.K(x) + b_0"
	coefficients[2+m] = 1; // We store the value of "alpha" from the main function to be solved, which is "y_hat = alpha.K(x) + b_0"
	
	// Free the Heap memory used for the locally allocated variables since they will no longer be used.
	free(X_tilde);
	free(TransposeOf_X_tilde);
	free(matMul1);
	free(matMul2);
	
	return;
}


/**
* The "trainPolynomialKernel()" is a static function that is
* used to apply the machine learning algorithm called Kernel
* machine classification with a polynomial Kernel. Within this
* process, the best fitting equation with the form of "y_hat =
* alpha.K(x) + b_0" will be identified with respect to the
* sampled data given through the argument pointer variables "X"
* and "Y". In that equation, "alpha" and "b_0" are the
* coefficients to be identified, "K(x)" is the Kernel function
* which is basically a transformation function of x, "alpha" is
* the coefficient of the Kernel, "b_0" is the bias coefficient
* of the model, "x" is a vector containing all the machine
* learning features and the coefficients within "K(x)" will be
* representend as "Beta". With this in mind and just like in
* the master thesis of Cesar Miranda Meza ("Machine learning to
* support applications with embedded systems and parallel
* computing"), the first step to train this model is to solve
* the kernel function. For this purpose and for this particular
* type of Kernel, a multiple polynomial regression will be
* applied with respect to the output of the system under study
* "Y". Both the characteristic equation of the multiple
* polynomial system and its best fitting coefficient values
* will together represent the Kernel function K(x). Now, for
* this particular Kernel machine classifier, because of being
* a polynomial system, the coefficient value of "alpha"=1 and
* "b_0"=0. As a result, when inserting all the coefficient
* values into this model, whenever its output is greater than
* the value of "0", it should be interpreted as the model
* predicting that the current input values represent group 1
* or the numeric value of "+1". Conversely, if the model
* produces a value less than "0", it should be interpreted as
* the model predicting that the current input values represent
* group 2 or the numeric value of "-1".
*
* NOTE: The algorithm section that applied the matrix inverse using
* the Gauss-Jordan method was inspired in the following source:
* "CodeSansar. Matrix Inverse Using Gauss Jordan Method C Program.
* November 16, 2021 (Recovery date), de CodeSansar Sitio web:
* https://bit.ly/3CowwSy".
*
* 
* @param float *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'float' MEMORY
*					 SPACES.
*
* @param float *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A SIZE OF
*					 "n" TIMES "p=1" 'float' MEMORY SPACES.
*					 Finally, make sure that the output values are
*					 either "+1" or "-1".
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param int N - This argument will represent the desired order of
*				 degree for the machine learning model to be trained.
*
* @param char isInteractionTerms = This argument variable will work as a
*						  		   flag to indicate whether the
*								   interaction terms are desired in the
*								   resulting model to be generated or
*								   not.
*						  		   Moreover, the possible valid values
*								   for this argument variable are:
*						  		   1) (int) 1 = The resulting model will
*												include the interaction
*												terms that are possible.
*												NOTE: This functionality,
*												is yet to be developed.
*						  		   2) (int) 0 = The resulting model will
*												not include interaction
*												terms.
*
* @param char isVariableOptimizer = This argument variable is not
*									having any effect on this function
*									at the moment, as its functionality
*									has not been developed. However, it
*									is recommended to initialize it with
*									an integer value of zero so that it
*									does not surprise the user when it
*									gets developed in the near future.
*
* @param float *coefficients - This argument will contain the pointer
*					 to a memory allocated variable in which we will
*					 store the identified best fitting coefficient
*					 values for the desired machine learning algorithm.
*					 These coefficients will each be stored in the same
*					 column but under different rows where the first
*					 coefficient (Beta_0) will be stored in the row with
*					 index 0, the last Beta value (Beta_{m*N}) will be
*					 stored in the row with index "{m*N}", the bias
*					 coefficient value (b_0) will be stored in the row
*					 with index "m*N+1" and the last coefficient (alpha)
*					 will be stored in the row with index "m*N+2". NOTE:
*					 For more information on how the "Beta" coefficients
*					 will be stored, please see the description given
*					 for the argument pointer variable "b" of the
*					 function "getMultiplePolynomialRegression()" in the
*					 "CenyMLregression.c" file. IT IS INDISPENSABLE THAT
*					 THIS VARIABLE IS ALLOCATED AND INITIALIZED WITH
*					 ZEROS BEFORE CALLING THIS FUNCTION WITH A VARIABLE
*					 SIZE OF "m*N+3" TIMES "p=1" 'float' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"coefficients".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 26, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
static void trainPolynomialKernel(float *X, float *Y, int n, int m, int p, int N, char isInteractionTerms, char isVariableOptimizer, float *coefficients) {
	// Determine whether the interaction terms are desired in the resulting model to be generated or not and then excecute the corresponding code.
	if (isInteractionTerms == 1) { // Include the interaction terms in the training process of the model to be generated.
		Serial.print("\nERROR: The functionality of this function, when the argument variable \"isInteractionTerms\" contains a value of 1, has not yet been developed.\n");
		exit(1);
		
		
	} else if (isInteractionTerms == 0) { // Do not inlcude the interaction terms in the training process of the model to be generated.		
		// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
		// Store the data that must be contained in the input matrix "X_tilde". In addition, we obtain the transpose of "X_tilde".
		int currentRowTimesmTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int mTimesNPlusOne = m*N+1; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
		int NplusOne = (N+1); //This variable is used to store a repetitive matheamtical operation, for performance purposes.
		float *X_tilde = (float *) malloc(n*mTimesNPlusOne*sizeof(float)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
		float *TransposeOf_X_tilde = (float *) malloc(mTimesNPlusOne*n*sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
		int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
		int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
		float increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
		for (int currentRow=0; currentRow<n; currentRow++) {
			currentRow2 = 0; // We reset the counters used in the following for-loop.
			currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
			currentRowTimesM = currentRow*m;
			X_tilde[currentRowTimesmTimesNplusOne] = 1;
			TransposeOf_X_tilde[currentColumn2] = X_tilde[currentRowTimesmTimesNplusOne];
			for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
				currentRowAndColumn = (currentColumn-1)*N + currentRowTimesmTimesNplusOne;
				increaseExponentialOfThisValue = 1;
				for (int currentExponential=1; currentExponential<NplusOne; currentExponential++) {
					currentRowAndColumn2 = currentExponential + currentRowAndColumn;
					increaseExponentialOfThisValue = increaseExponentialOfThisValue * X[currentColumn-1 + currentRowTimesM];
					X_tilde[currentRowAndColumn2] = increaseExponentialOfThisValue;
					currentRow2++;
					TransposeOf_X_tilde[currentColumn2 + currentRow2*n] = X_tilde[currentRowAndColumn2];
				}
			}
			currentColumn2++;
		}
		
		// -------------------- SOLUTION OF THE MODEL -------------------- //
		// NOTE: To solve for the "Beta_0", "Beta_1", ..., "Beta_m*N" coefficients, we will apply a multiple polynomial regression.
		// In order to start obtaining the coefficients, we multiply the matrix "TransposeOf_X_tilde" with the matrix "X_tilde" and store the result in the matrix "matMul1".
		int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int currentColumnTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		float *matMul1 = (float *) calloc(mTimesNPlusOne*mTimesNPlusOne, sizeof(float)); // We allocate, and initialize with zeros, the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "X_tilde" and its transpose.
		for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) {
			currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
			currentRowTimesN = currentRow*n;
			for (int currentColumn=0; currentColumn<mTimesNPlusOne; currentColumn++) {
				currentColumnTimesN = currentColumn*n;
				currentRowAndColumn = currentColumn + currentRowTimesmTimesNplusOne;
				for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
					// Here we want to multiply "TransposeOf_X_tilde" with the matrix "X_tilde", but we will use "TransposeOf_X_tilde" for such multiplication since they contain the same data, for performance purposes.
					matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesN] * TransposeOf_X_tilde[currentMultipliedElements + currentColumnTimesN];
				}
		    }
		}
		
		// In order to continue obtaining the coefficients, we innitialize the data of a unitary matrix with the same dimensions of the matrix "matMul1".
		// NOTE: Because getting the data of the transpose of "X_tilde"
		//		 directly from that same variable ("X_tilde"), will
		//		 increase performance in further steps, we will store the
		//		 matrix inverse of "matMul1" in the variable
		//		 "TransposeOf_X_tilde", in order to maximize computational
		//		 resources and further increase performance by not having to
		//		 allocate more memory in the computer system.
		for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) { // We set all values to zero.
			currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
			for (int currentColumn=0; currentColumn<mTimesNPlusOne; currentColumn++) {
				TransposeOf_X_tilde[currentColumn + currentRowTimesmTimesNplusOne] = 0;
		    }
		}
		for (int currentUnitaryValue=0; currentUnitaryValue<mTimesNPlusOne; currentUnitaryValue++) { // We set the corresponding 1's values to make the corresponding unitary matrix.
			TransposeOf_X_tilde[currentUnitaryValue + currentUnitaryValue*mTimesNPlusOne] = 1;
		}
		
		// In order to continue obtaining the coefficients, we calculate the matrix inverse of "matMul1" with the Gauss-Jordan approach and store its result in "TransposeOf_X_tilde".
		int currentColumnTimesmTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		float ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
		for (int currentColumn=0; currentColumn<mTimesNPlusOne; currentColumn++) { // We apply the differentiations applied to each row according to the approach used.
			currentColumnTimesmTimesNplusOne = currentColumn*mTimesNPlusOne;
			currentRowAndColumn2 = currentColumn + currentColumnTimesmTimesNplusOne;
			for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) {
				if (currentRow != currentColumn) {
					currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
					ratioModifier = matMul1[currentColumn + currentRowTimesmTimesNplusOne]/matMul1[currentRowAndColumn2];
					for (int currentModifiedElements=0; currentModifiedElements<mTimesNPlusOne; currentModifiedElements++) { // We apply the current process to the principal matrix.
						currentRowAndColumn = currentModifiedElements + currentRowTimesmTimesNplusOne;
						matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] - ratioModifier * matMul1[currentModifiedElements + currentColumnTimesmTimesNplusOne];
					}
					for (int currentModifiedElements=0; currentModifiedElements<mTimesNPlusOne; currentModifiedElements++) { // We apply the current process to the result matrix.
						currentRowAndColumn = currentModifiedElements + currentRowTimesmTimesNplusOne;
						TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] - ratioModifier * TransposeOf_X_tilde[currentModifiedElements + currentColumnTimesmTimesNplusOne];
					}
				}
			}
	    }
		for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) { // We apply the last step of the approach used in order to obtain the diagonal of 1's in the principal matrix.
			currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
			currentRowAndColumn2 = currentRow + currentRowTimesmTimesNplusOne;
			for (int currentColumn=0; currentColumn<mTimesNPlusOne; currentColumn++) {
				currentRowAndColumn = currentColumn + currentRowTimesmTimesNplusOne;
				TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] / matMul1[currentRowAndColumn2];
			}
	    }
	    
		// In order to continue obtaining the coefficients, we multiply the inverse matrix that was obtained by the transpose of the matrix "X_tilde".
		// NOTE: Remember that we will get the data of the transpose of
		//		 "X_tilde" directly from that same variable
		//		 ("X_tilde") due to performance reasons and; that the
		//		 inverse matrix that was obtained is stored in
		//		 "TransposeOf_X_tilde".
		float *matMul2 = (float *) calloc(mTimesNPlusOne*n, sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "X_tilde".
		for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) {
			currentRowTimesN = currentRow*n;
			currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
			for (int currentColumn=0; currentColumn<n; currentColumn++) {
				currentRowAndColumn = currentColumn + currentRowTimesN;
				currentColumnTimesmTimesNplusOne = currentColumn*mTimesNPlusOne;
				for (int currentMultipliedElements=0; currentMultipliedElements<mTimesNPlusOne; currentMultipliedElements++) {
					matMul2[currentRowAndColumn] = matMul2[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesmTimesNplusOne] * X_tilde[currentMultipliedElements + currentColumnTimesmTimesNplusOne];
				}
		    }
		}
		
		// In order to conclude obtaining the coefficients ("coefficients"), we multiply the previously resulting matrix ("matMul2") by the output matrix "Y".
		for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) { // In this for-loop, we will store the "Beta" coefficient values of the Kernel function "K(x)".
			currentRowTimesN = currentRow*n;
			for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
				coefficients[currentRow] = coefficients[currentRow] + matMul2[currentMultipliedElements + currentRowTimesN] * Y[currentMultipliedElements];
			}
		}
		coefficients[1+m*N] = 0; // We store the value of "b_0" from the main function to be solved, which is "y_hat = alpha.K(x) + b_0"
		coefficients[2+m*N] = 1; // We store the value of "alpha" from the main function to be solved, which is "y_hat = alpha.K(x) + b_0"
		
		// Free the Heap memory used for the locally allocated variables since they will no longer be used.
		free(X_tilde);
		free(TransposeOf_X_tilde);
		free(matMul1);
		free(matMul2);
		
		
	} else { // The argument variable "isInteractionTerms" has been assigned an invalid value. Therefore, inform the user about this and terminate the program.
		Serial.print("\nERROR: The argument variable \"isInteractionTerms\" is meant to store only a binary value that equals either 1 or 0.\n");
		exit(1);
	}
	
	return;
}


/**
* The "trainLogisticKernel()" is a static function that is used
* to apply the machine learning algorithm called Kernel machine
* classification with a logistic Kernel. Within this process, the
* best fitting equation with the form of "y_hat = alpha.K(x) +
* b_0" will be identified with respect to the sampled data given
* through the argument pointer variables "X" and "Y". In that
* equation, "alpha" and "b_0" are the coefficients to be
* identified, "K(x)" is the Kernel function which is basically a
* transformation function of x, "alpha" is the coefficient of the
* Kernel, "b_0" is the bias coefficient of the model, "x" is a
* vector containing all the machine learning features and the
* coefficients within "K(x)" will be representend as "Beta". With
* this in mind and just like in the master thesis of Cesar
* Miranda Meza ("Machine learning to support applications with
* embedded systems and parallel computing"), the first step to
* train this model is to solve the kernel function. For this
* purpose and for this particular type of Kernel, a multiple
* linear regression will be applied with respect to the required
* transformed output of the system under study "Y_tilde" in order
* to obtain the equivalent of a logistic regression (see the
* thesis of Cesar Miranda Meza for more details). Both the
* characteristic equation of the logistic system and its best
* fitting coefficient values will together represent the Kernel
* function K(x). The next step is to now apply the transformation
* of the Kernel function "K(x)" with the coefficients that were
* obtained to then store its output. Finally, we will use the
* output of "K(x)" as the input data of a simple linear
* regression in order to learn the best fitting coefficient
* values of "alpha" and "b_0". As a result, when inserting all
* the coefficient values into the main model which is
* "y_hat = alpha.K(x) + b_0", whenever its output is greater than
* the value of "0", it should be interpreted as the model
* predicting that the current input values represent group 1 or
* the numeric value of "+1". Conversely, if the model produces a
* value less than "0", it should be interpreted as the model
* predicting that the current input values represent group 2 or
* the numeric value of "-1".
*
* NOTE: The algorithm section that applied the matrix inverse using
* the Gauss-Jordan method was inspired in the following source:
* "CodeSansar. Matrix Inverse Using Gauss Jordan Method C Program.
* November 16, 2021 (Recovery date), de CodeSansar Sitio web:
* https://bit.ly/3CowwSy".
*
* 
* @param float *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'float' MEMORY
*					 SPACES.
*
* @param float *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A SIZE OF
*					 "n" TIMES "p=1" 'float' MEMORY SPACES.
*					 Finally, make sure that the output values are
*					 either "+1" or "-1".
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param char isVariableOptimizer = This argument variable is not
*									having any effect on this function
*									at the moment, as its functionality
*									has not been developed. However, it
*									is recommended to initialize it with
*									an integer value of zero so that it
*									does not surprise the user when it
*									gets developed in the near future.
*
* @param float *coefficients - This argument will contain the pointer
*					 to a memory allocated variable in which we will
*					 store the identified best fitting coefficient values
*					 for the desired machine learning algorithm. These
*					 coefficients will each be stored in the same
*					 column but under different rows where the first
*					 coefficient (Beta_0) will be stored in the row with
*					 index 0, the last Beta value (Beta_m) will be
*					 stored in the row with index "m", the bias
*					 coefficient value (b_0) will be stored in the row
*					 with index "m+1" and the last coefficient (alpha)
*					 will be stored in the row with index "m+2". NOTE:
*					 For more information on how the "Beta" coefficients
*					 will be stored, please see the description given for
*					 the argument pointer variable "coefficients" of the
*					 function "getLogisticRegression()" in the
*					 "CenyMLregression.c" file. IT IS INDISPENSABLE THAT
*					 THIS VARIABLE IS ALLOCATED AND INITIALIZED WITH
*					 ZEROS BEFORE CALLING THIS FUNCTION WITH A VARIABLE
*					 SIZE OF "m+3" TIMES "p=1" 'float' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"coefficients".
*
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 27, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
static void trainLogisticKernel(float *X, float *Y, int n, int m, int p, char isVariableOptimizer, float *coefficients) {
	// Store the data that must be contained in the input matrix "X_tilde". In addition, we obtain the transpose of "X_tilde".
	int currentRowTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	float *Y_tilde = (float *) malloc(n*p*sizeof(float)); // This variable will contain the output data of the system under study ("Y") as required by the training of this algorithm.
	float *X_tilde = (float *) malloc(n*mPlusOne*sizeof(float)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
	float *TransposeOf_X_tilde = (float *) malloc(mPlusOne*n*sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
	for (int currentRow=0; currentRow<n; currentRow++) {
		// --------------- PREPROCESSING OF THE OUTPUT DATA --------------- //
		if (Y[currentRow] == 1) {
			Y_tilde[currentRow] = 0.9999;
		}
		if (Y[currentRow] == -1) {
			Y_tilde[currentRow] = 0.0001;
		}
		Y_tilde[currentRow] = log(Y_tilde[currentRow]/(1-Y_tilde[currentRow]));
		
		// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
		currentRow2 = 0; // We reset the counters used in the following for-loop.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowTimesM = currentRow*m;
		X_tilde[currentRowTimesMplusOne] = 1;
		TransposeOf_X_tilde[currentColumn2] = X_tilde[currentRowTimesMplusOne];
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			X_tilde[currentRowAndColumn] = X[currentColumn-1 + currentRowTimesM];
			currentRow2++;
			TransposeOf_X_tilde[currentColumn2 + currentRow2*n] = X_tilde[currentRowAndColumn];
		}
		currentColumn2++;
	}
	
	// -------------------- SOLUTION OF THE KERNEL -------------------- //
	// NOTE: To solve for the "Beta_0", "Beta_1", ..., "Beta_m" coefficients, we will apply a multiple linear regression.
	// In order to start obtaining the coefficients, we multiply the matrix "TransposeOf_X_tilde" with the matrix "X_tilde" and store the result in the matrix "matMul1".
	int currentColumnTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentColumnTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	float *matMul1 = (float *) calloc(mPlusOne*mPlusOne, sizeof(float)); // We allocate, and initialize with zeros, the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "X_tilde" and its transpose.
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowTimesN = currentRow*n;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			currentColumnTimesN = currentColumn*n;
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
				// Here we want to multiply "TransposeOf_X_tilde" with the matrix "X_tilde", but we will use "TransposeOf_X_tilde" for such multiplication since they contain the same data, for performance purposes.
				matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesN] * TransposeOf_X_tilde[currentMultipliedElements + currentColumnTimesN];
			}
	    }
	}
	
	// In order to continue obtaining the coefficients, we innitialize the data of a unitary matrix with the same dimensions of the matrix "matMul1".
	// NOTE: Because getting the data of the transpose of "X_tilde"
	//		 directly from that same variable ("X_tilde"), will
	//		 increase performance in further steps, we will store the
	//		 matrix inverse of "matMul1" in the variable
	//		 "TransposeOf_X_tilde", in order to maximize computational
	//		 resources and further increase performance by not having to
	//		 allocate more memory in the computer system.
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // We set all values to zero.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			TransposeOf_X_tilde[currentColumn + currentRowTimesMplusOne] = 0;
	    }
	}
	for (int currentUnitaryValue=0; currentUnitaryValue<mPlusOne; currentUnitaryValue++) { // We set the corresponding 1's values to make the corresponding unitary matrix.
		TransposeOf_X_tilde[currentUnitaryValue + currentUnitaryValue*mPlusOne] = 1;
	}
	
	// In order to continue obtaining the coefficients, we calculate the matrix inverse of "matMul1" with the Gauss-Jordan approach and store its result in "TransposeOf_X_tilde".
	int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	float ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
	for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) { // We apply the differentiations applied to each row according to the approach used.
		currentColumnTimesMplusOne = currentColumn*mPlusOne;
		currentRowAndColumn2 = currentColumn + currentColumnTimesMplusOne;
		for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
			if (currentRow != currentColumn) {
				currentRowTimesMplusOne = currentRow*mPlusOne;
				ratioModifier = matMul1[currentColumn + currentRowTimesMplusOne]/matMul1[currentRowAndColumn2];
				for (int currentModifiedElements=0; currentModifiedElements<mPlusOne; currentModifiedElements++) { // We apply the current process to the principal matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesMplusOne;
					matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] - ratioModifier * matMul1[currentModifiedElements + currentColumnTimesMplusOne];
				}
				for (int currentModifiedElements=0; currentModifiedElements<mPlusOne; currentModifiedElements++) { // We apply the current process to the result matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesMplusOne;
					TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] - ratioModifier * TransposeOf_X_tilde[currentModifiedElements + currentColumnTimesMplusOne];
				}
			}
		}
    }
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // We apply the last step of the approach used in order to obtain the diagonal of 1's in the principal matrix.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowAndColumn2 = currentRow + currentRowTimesMplusOne;
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] / matMul1[currentRowAndColumn2];
		}
    }
    
	// In order to continue obtaining the coefficients, we multiply the inverse matrix that was obtained by the transpose of the matrix "X_tilde".
	// NOTE: Remember that we will get the data of the transpose of
	//		 "X_tilde" directly from that same variable
	//		 ("X_tilde") due to performance reasons and; that the
	//		 inverse matrix that was obtained is stored in
	//		 "TransposeOf_X_tilde".
	float *matMul2 = (float *) calloc(mPlusOne*n, sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "X_tilde".
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
		currentRowTimesN = currentRow*n;
		currentRowTimesMplusOne = currentRow*mPlusOne;
		for (int currentColumn=0; currentColumn<n; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesN;
			currentColumnTimesMplusOne = currentColumn*mPlusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<mPlusOne; currentMultipliedElements++) {
				matMul2[currentRowAndColumn] = matMul2[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesMplusOne] * X_tilde[currentMultipliedElements + currentColumnTimesMplusOne];
			}
	    }
	}
	
	// In order to conclude obtaining the coefficients ("coefficients"), we multiply the previously resulting matrix ("matMul2") by the output matrix "Y_tilde".
	for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // In this for-loop, we will store the "Beta" coefficient values of the Kernel function "K(x)".
		currentRowTimesN = currentRow*n;
		for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
			coefficients[currentRow] = coefficients[currentRow] + matMul2[currentMultipliedElements + currentRowTimesN] * Y_tilde[currentMultipliedElements];
		}
	}
	
	// We declare and innitialize the variables that will be required to calculate the coefficients of "alpha" and "b_0".
	// NOTE: To solve for the "alpha" and "b_0" coefficients, we will apply a simple linear regression.
	float sumOf_xy = 0;
	float sumOf_y = 0;
	float sumOf_x = 0;
	float sumOf_xSquared = 0;
	for (int currentRow = 0; currentRow < n; currentRow++) {
		// -------------------- APLICATION OF THE KERNEL FUNCTION -------------------- //
		// We predict all the requested input values (X) with the machine learning model that was just obtained, which represents the Kernel function applied to the input data.
		Y_tilde[currentRow] = coefficients[0];
		currentRowTimesM = currentRow*m;
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			Y_tilde[currentRow] = Y_tilde[currentRow] + coefficients[currentColumn]*X[currentColumn-1 + currentRowTimesM];
		}
		Y_tilde[currentRow] = 1 / (1 + exp(-Y_tilde[currentRow])); // We will store the results of the transformed function "K(x)" in "Y_tilde".
		
		// -------------------- SOLUTION OF "alpha" AND "b_0" -------------------- //
		// In order to obtain the coefficients of "alpha" and "b_0", we apply a linear regression using the input data "Y_tilde" (that was obtained with the transformed function "K(x)") with respect to the output data "Y".
		sumOf_xy += Y_tilde[currentRow] * Y[currentRow];
		sumOf_y += Y[currentRow];
		sumOf_x += Y_tilde[currentRow];
		sumOf_xSquared += Y_tilde[currentRow] * Y_tilde[currentRow];
	}
	
	// We calculate the value of "alpha" from the main function to be solved, which is "y_hat = alpha.K(x) + b_0"
	coefficients[2+m] = (n*sumOf_xy - sumOf_y*sumOf_x)/(n*sumOf_xSquared - sumOf_x*sumOf_x);
	
	// We calculate the value of "b_0" from the main function to be solved, which is "y_hat = alpha.K(x) + b_0"
	coefficients[1+m] = (sumOf_y - coefficients[2+m]*sumOf_x)/n;
	
	// Free the Heap memory used for the locally allocated variables since they will no longer be used.
	free(Y_tilde);
	free(X_tilde);
	free(TransposeOf_X_tilde);
	free(matMul1);
	free(matMul2);
	return;
}


/**
* The "trainGaussianKernel()" is a static function that is used
* to apply the machine learning algorithm called Kernel machine
* classification with a gaussian Kernel. Within this process, the
* best fitting equation with the form of "y_hat = alpha.K(x) +
* b_0" will be identified with respect to the sampled data given
* through the argument pointer variables "X" and "Y". In that
* equation, "alpha" and "b_0" are the coefficients to be
* identified, "K(x)" is the Kernel function which is basically a
* transformation function of x, "alpha" is the coefficient of the
* Kernel, "b_0" is the bias coefficient of the model, "x" is a
* vector containing all the machine learning features and the
* coefficients within "K(x)" will be representend as "Beta". With
* this in mind and just like in the master thesis of Cesar
* Miranda Meza ("Machine learning to support applications with
* embedded systems and parallel computing"), the first step to
* train this model is to solve the kernel function. For this
* purpose and for this particular type of Kernel, a multiple
* polynomial regression will be applied with respect to the
* required transformed output of the system under study
* "Y_tilde" in order to obtain the equivalent of a gaussian
* regression (see the thesis of Cesar Miranda Meza for more
* details). Both the characteristic equation of the gaussian
* system and its best fitting coefficient values will together
* represent the Kernel function K(x). The next step is to now
* apply the transformation of the Kernel function "K(x)" with
* the coefficients that were obtained to then store its output.
* Finally, we will use the output of "K(x)" as the input data of
* a simple linear regression in order to learn the best fitting
* coefficient values of "alpha" and "b_0". As a result, when
* inserting all the coefficient values into the main model which
* is "y_hat = alpha.K(x) + b_0", whenever its output is greater
* than the value of "0", it should be interpreted as the model
* predicting that the current input values represent group 1 or
* the numeric value of "+1". Conversely, if the model produces a
* value less than "0", it should be interpreted as the model
* predicting that the current input values represent group 2 or
* the numeric value of "-1".
*
* NOTE: The algorithm section that applied the matrix inverse using
* the Gauss-Jordan method was inspired in the following source:
* "CodeSansar. Matrix Inverse Using Gauss Jordan Method C Program.
* November 16, 2021 (Recovery date), de CodeSansar Sitio web:
* https://bit.ly/3CowwSy".
*
* 
* @param float *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'float' MEMORY
*					 SPACES.
*
* @param float *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A SIZE OF
*					 "n" TIMES "p=1" 'float' MEMORY SPACES.
*					 Finally, make sure that the output values are
*					 either "+1" or "-1".
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param float zeroEpsilon - This argument will contain the user defined
*							  epsilon value that will be used to temporarly
*							  store any "0", that is contained in the
*							  output matrix "Y", with the added value of
*							  "zeroEpsilon" but only for the gaussian
*							  regression training process. The process in
*							  which the "zeroEpsilon" is part of, is a
*							  strictly needed mathematical operation in the
*							  calculus of the output transformation
*							  "Y_tilde = ln(Y)" of the mentioned regression
*							  algorithm. The restriction that has to be
*							  complied is "0 < zeroEpsilon < 1" and its
*							  value will change the bias of the gaussian
*							  Kernel. Ussually a good value is
*							  "zeroEpsilon = 0.1" but other values can be
*							  tried in order to better fit the model.
*							  IMPORTANT NOTE: The results will be
*							  temporarly stored so that the values of the
*							  output matrix "Y" is not modified.
*
* @param char isForceGaussianCurve = This argument variable will
*							determine the types of coefficients to be
*							obtained during the training process. If
*							"isForceGaussianCurve" = 0, then the
*							coefficients to be obtained will be
*							directly from a model that has an
*							exponential with the multiple polynomial
*							equation without its interaction terms from
*							the following mathematical form "y_hat =
*							exp( b_tilde_{0} + b_tilde_{1}*x_{1} +
*							b_tilde_{2}*x_{1}^2 + b_tilde_{3}*x_{2} +
* 							b_tilde_{4}*x_{2}^2 + ... +
*							b_tilde_{2*m-1}*x_{m} +
* 							b_tilde_{2*m}*x_{m}^2 )".
*							If "isForceGaussianCurve" = 1, then the
*							coefficients to be obtained will be in the
*							from of an approximation from the previous
*							equation, such that each machine learning
*							feature will have a particular "mean" and
*							"variance" coefficient values. These will be
*							used to govern the following model instead
*							"y_hat =
*							exp(-( ((x_{1}-mean_{1})^2)/2*variance_{1} +
* 							((x_{2}-mean_{2})^2)/2*variance_{2} + ... +
* 							((x_{m}-mean_{m})^2)/2*variance_{m} ))".
*							On the other hand, have in consideration
*							that because the solution of the multiple
*							polynomial equation can sometimes give a non
*							perfect square binomial, when
*							"isForceGaussianCurve" = 0, you may not get
*							a true gaussian form under such circumstance.
*							Therefore, by setting "isForceGaussianCurve"
*							= 1, you will force the solution to always
*							have a gaussian form but this may increase
*							the total obtained error as a consequence.
*
* @param char isVariableOptimizer = This argument variable is not
*									having any effect on this function
*									at the moment, as its functionality
*									has not been developed. However, it
*									is recommended to initialize it with
*									an integer value of zero so that it
*									does not surprise the user when it
*									gets developed in the near future.
*
* @param float *coefficients - This argument will contain the pointer
*					 to a memory allocated variable in which we will
*					 store the identified best fitting coefficient values
*					 for the desired machine learning algorithm.  IT IS
*					 INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED AND
*					 INITIALIZED WITH ZEROS BEFORE CALLING THIS FUNCTION
*					 WITH A VARIABLE SIZE OF "m*2+3" TIMES "p=1" 'float'
*					 MEMORY SPACES. Moreover, the coefficients will be
*					 stored in a different manner depending on the value
*					 of the argument variable "isForceGaussianCurve". If
*					 its value equals 0, the coefficients will each be
*					 stored in the same column but under different rows
*					 where the first coefficient (Beta_0) will be stored
*					 in the row with index 0, the last Beta value
*					 (Beta_{2*m}) will be stored in the row with index
*					 "2*m", the bias coefficient value (b_0) will be
*					 stored in the row with index "2*m+1" and the last
*					 coefficient (alpha) will be stored in the row with
*					 index "2*m+2". On the other hand, if
*					 "isForceGaussianCurve" = 1, then each machine
*					 learning feature will contain two coefficients but
*					 only in the Kernel function. The first will be the
*					 "mean" and the second will be the "variance". Both
*					 of these will be stored in the same row, where the
*					 first column (column index 0) will store the mean and
*					 the variance will be stored in the second column
*					 (column index 1). In addition, each of the identified
*					 means and variances for each different machine
*					 learning feature will be stored separately in rows.
*					 The first row (row index 0) will contain the mean and
*					 variance for the first machine learning feature and
*					 the last row (row index "m") will contain the mean
*					 and variance for the "m"-th machine learning feature.
*					 Furthermore, note that if "isForceGaussianCurve" = 1,
*					 then the true size to be taken into account of the
*					 argument pointer variable "coefficients" will be from
*					 "m*2+3" to "m*2+2". This means that the last index
*					 with respect to the entirely allocated memory in
*					 "coefficients" will not be used or have any meaning
*					 under this case. Finally, the bias coefficient value
*					 (b_0) will be stored in the row with index "m*2" and
*					 column index "0" and the last coefficient (alpha)
*					 will be stored in the row with index "m*2+1" and
*					 column index "1".
*				     NOTE: For more information on how the "Beta"
*					 coefficients will be stored, please see the
*					 description given for the argument pointer variable
*					 "b" of the function "getLogisticRegression()" in the
*					 "CenyMLregression.c" file.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"coefficients".
*
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 27, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
static void trainGaussianKernel(float *X, float *Y, int n, int m, int p, float zeroEpsilon, char isForceGaussianCurve, char isVariableOptimizer, float *coefficients) {
	// If the argument flag variable "isForceGaussianCurve" is different than the value of "1" and "0", then emit an error message and terminate the program. Otherwise, continue with the program.
	if (isForceGaussianCurve != 1) {
		if (isForceGaussianCurve != 0) {
			Serial.print("\nERROR: Please assign a valid value for the flag \"isForceGaussianCurve\", which may be 1 or 0.\n");
			exit(1);
		}
	}
	
	// Store the data that must be contained in the input matrix "X_tilde". In addition, we obtain the transpose of "X_tilde".
	int N = 2; // This variable is used to store the desired order of degree for the machine learning model to be trained.
	int currentRowTimesmTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int mTimesNPlusOne = m*N+1; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	int NplusOne = (N+1); //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	float *Y_tilde = (float *) malloc(n*p*sizeof(float)); // This variable will contain the output data of the system under study ("Y") as required by the training of this algorithm.
	float *X_tilde = (float *) malloc(n*mTimesNPlusOne*sizeof(float)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
	float *TransposeOf_X_tilde = (float *) malloc(mTimesNPlusOne*n*sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
	float increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
	for (int currentRow=0; currentRow<n; currentRow++) {
		// --------------- PREPROCESSING OF THE OUTPUT DATA --------------- //
		if (Y[currentRow] == 1) {
			Y_tilde[currentRow] = log(1);
		} else if (Y[currentRow] == -1) {
			Y_tilde[currentRow] = log(zeroEpsilon);
		}
		
		// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
		currentRow2 = 0; // We reset the counters used in the following for-loop.
		currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
		currentRowTimesM = currentRow*m;
		X_tilde[currentRowTimesmTimesNplusOne] = 1;
		TransposeOf_X_tilde[currentColumn2] = X_tilde[currentRowTimesmTimesNplusOne];
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = (currentColumn-1)*N + currentRowTimesmTimesNplusOne;
			increaseExponentialOfThisValue = 1;
			for (int currentExponential=1; currentExponential<NplusOne; currentExponential++) {
				currentRowAndColumn2 = currentExponential + currentRowAndColumn;
				increaseExponentialOfThisValue = increaseExponentialOfThisValue * X[currentColumn-1 + currentRowTimesM];
				X_tilde[currentRowAndColumn2] = increaseExponentialOfThisValue;
				currentRow2++;
				TransposeOf_X_tilde[currentColumn2 + currentRow2*n] = X_tilde[currentRowAndColumn2];
			}
		}
		currentColumn2++;
	}
	
	// -------------------- SOLUTION OF THE KERNEL -------------------- //
	// In order to start obtaining the coefficients, we multiply the matrix "TransposeOf_X_tilde" with the matrix "X_tilde" and store the result in the matrix "matMul1".
	int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentColumnTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	float *matMul1 = (float *) calloc(mTimesNPlusOne*mTimesNPlusOne, sizeof(float)); // We allocate, and initialize with zeros, the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "X_tilde" and its transpose.
	for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) {
		currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
		currentRowTimesN = currentRow*n;
		for (int currentColumn=0; currentColumn<mTimesNPlusOne; currentColumn++) {
			currentColumnTimesN = currentColumn*n;
			currentRowAndColumn = currentColumn + currentRowTimesmTimesNplusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
				// Here we want to multiply "TransposeOf_X_tilde" with the matrix "X_tilde", but we will use "TransposeOf_X_tilde" for such multiplication since they contain the same data, for performance purposes.
				matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesN] * TransposeOf_X_tilde[currentMultipliedElements + currentColumnTimesN];
			}
	    }
	}
	
	// In order to continue obtaining the coefficients, we innitialize the data of a unitary matrix with the same dimensions of the matrix "matMul1".
	// NOTE: Because getting the data of the transpose of "X_tilde"
	//		 directly from that same variable ("X_tilde"), will
	//		 increase performance in further steps, we will store the
	//		 matrix inverse of "matMul1" in the variable
	//		 "TransposeOf_X_tilde", in order to maximize computational
	//		 resources and further increase performance by not having to
	//		 allocate more memory in the computer system.
	for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) { // We set all values to zero.
		currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
		for (int currentColumn=0; currentColumn<mTimesNPlusOne; currentColumn++) {
			TransposeOf_X_tilde[currentColumn + currentRowTimesmTimesNplusOne] = 0;
	    }
	}
	for (int currentUnitaryValue=0; currentUnitaryValue<mTimesNPlusOne; currentUnitaryValue++) { // We set the corresponding 1's values to make the corresponding unitary matrix.
		TransposeOf_X_tilde[currentUnitaryValue + currentUnitaryValue*mTimesNPlusOne] = 1;
	}
	
	// In order to continue obtaining the coefficients, we calculate the matrix inverse of "matMul1" with the Gauss-Jordan approach and store its result in "TransposeOf_X_tilde".
	int currentColumnTimesmTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	float ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
	for (int currentColumn=0; currentColumn<mTimesNPlusOne; currentColumn++) { // We apply the differentiations applied to each row according to the approach used.
		currentColumnTimesmTimesNplusOne = currentColumn*mTimesNPlusOne;
		currentRowAndColumn2 = currentColumn + currentColumnTimesmTimesNplusOne;
		for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) {
			if (currentRow != currentColumn) {
				currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
				ratioModifier = matMul1[currentColumn + currentRowTimesmTimesNplusOne]/matMul1[currentRowAndColumn2];
				for (int currentModifiedElements=0; currentModifiedElements<mTimesNPlusOne; currentModifiedElements++) { // We apply the current process to the principal matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesmTimesNplusOne;
					matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] - ratioModifier * matMul1[currentModifiedElements + currentColumnTimesmTimesNplusOne];
				}
				for (int currentModifiedElements=0; currentModifiedElements<mTimesNPlusOne; currentModifiedElements++) { // We apply the current process to the result matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesmTimesNplusOne;
					TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] - ratioModifier * TransposeOf_X_tilde[currentModifiedElements + currentColumnTimesmTimesNplusOne];
				}
			}
		}
    }
	for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) { // We apply the last step of the approach used in order to obtain the diagonal of 1's in the principal matrix.
		currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
		currentRowAndColumn2 = currentRow + currentRowTimesmTimesNplusOne;
		for (int currentColumn=0; currentColumn<mTimesNPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesmTimesNplusOne;
			TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] / matMul1[currentRowAndColumn2];
		}
    }
    
	// In order to continue obtaining the coefficients, we multiply the inverse matrix that was obtained by the transpose of the matrix "X_tilde".
	// NOTE: Remember that we will get the data of the transpose of
	//		 "X_tilde" directly from that same variable
	//		 ("X_tilde") due to performance reasons and; that the
	//		 inverse matrix that was obtained is stored in
	//		 "TransposeOf_X_tilde".
	float *matMul2 = (float *) calloc(mTimesNPlusOne*n, sizeof(float)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "X_tilde".
	for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) {
		currentRowTimesN = currentRow*n;
		currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
		for (int currentColumn=0; currentColumn<n; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesN;
			currentColumnTimesmTimesNplusOne = currentColumn*mTimesNPlusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<mTimesNPlusOne; currentMultipliedElements++) {
				matMul2[currentRowAndColumn] = matMul2[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesmTimesNplusOne] * X_tilde[currentMultipliedElements + currentColumnTimesmTimesNplusOne];
			}
	    }
	}
	
	// In order to conclude obtaining the coefficients ("b_tilde"), we multiply the previously resulting matrix ("matMul2") by the output matrix "Y".
	for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) {
		currentRowTimesN = currentRow*n;
		for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
			coefficients[currentRow] = coefficients[currentRow] + matMul2[currentMultipliedElements + currentRowTimesN] * Y_tilde[currentMultipliedElements];
		}
	}
	
	// Finally, if the flag "isForceGaussianCurve" is set, then we use the resulting coefficients data ("b_tilde") to obtain an approximation of the variance and the mean values that are part of the gaussian equation. This is because the previously obtained coefficients ("b_tilde") are transformed values (see the mathematical formulation for more details).
	// NOTE: If the flag "isForceGaussianCurve" is not set, then the complete coefficients of the multiple polynomial solution will be returned instead.
	if (isForceGaussianCurve == 1) {
		int currentRowTimesTwo; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int onePlusCurrentRowTimesTwo; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		for (int currentRow=0; currentRow<m; currentRow++) {
			currentRowTimesTwo = currentRow*2;
			onePlusCurrentRowTimesTwo = 1 + currentRowTimesTwo;
			coefficients[currentRowTimesTwo] = coefficients[onePlusCurrentRowTimesTwo];
			coefficients[onePlusCurrentRowTimesTwo] = -1/(2*coefficients[2 + currentRowTimesTwo]); // We obtain and store the approximation/expected value of the variance coefficient value of the current machine learning feature.
			coefficients[currentRowTimesTwo] = coefficients[currentRowTimesTwo] * coefficients[onePlusCurrentRowTimesTwo]; // We obtain and store the approximation/expected value of the mean coefficient value of the current machine learning feature.
		}
	}
	
	// We declare and innitialize the variables that will be required to calculate the coefficients of "alpha" and "b_0".
	// NOTE: To solve for the "alpha" and "b_0" coefficients, we will apply a simple linear regression.
	float sumOf_xy = 0;
	float sumOf_y = 0;
	float sumOf_x = 0;
	float sumOf_xSquared = 0;
	// We predict all the requested input values (X) with the machine learning model that was just obtained, which represents the Kernel function applied to the input data.
	if (isForceGaussianCurve == 1) { // The given coefficient values that were obtained for the Kernel are the literal mean and variance.
		// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (coefficients).
		float squareThisValue; // Variable used to store the value that wants to be squared.
		int currentColumnTimesTwo; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		for (int currentRow=0; currentRow<n; currentRow++) {
			// -------------------- APLICATION OF THE KERNEL FUNCTION -------------------- //
			Y_tilde[currentRow] = 0;
			for (int currentColumn=0; currentColumn<m; currentColumn++) {
				currentColumnTimesTwo = currentColumn*2;
				squareThisValue = X[currentColumn + currentRow*m] - coefficients[currentColumnTimesTwo];
				Y_tilde[currentRow] = Y_tilde[currentRow] + squareThisValue * squareThisValue / (2*coefficients[1 + currentColumnTimesTwo]);
			}
			Y_tilde[currentRow] = exp(-Y_tilde[currentRow]); // We will store the results of the transformed function "K(x)" in "Y_tilde".
			
			// -------------------- SOLUTION OF "alpha" AND "b_0" -------------------- //
			// In order to obtain the coefficients of "alpha" and "b_0", we apply a linear regression using the input data "Y_tilde" (that was obtained with the transformed function "K(x)") with respect to the output data "Y_tilde".
			sumOf_xy += Y_tilde[currentRow] * Y[currentRow];
			sumOf_y += Y[currentRow];
			sumOf_x += Y_tilde[currentRow];
			sumOf_xSquared += Y_tilde[currentRow] * Y_tilde[currentRow];
		}
		
		// We calculate the value of "alpha" from the main function to be solved, which is "y_hat = alpha.K(x) + b_0"
		coefficients[1+m*2] = (n*sumOf_xy - sumOf_y*sumOf_x)/(n*sumOf_xSquared - sumOf_x*sumOf_x);
		
		// We calculate the value of "b_0" from the main function to be solved, which is "y_hat = alpha.K(x) + b_0"
		coefficients[m*2] = (sumOf_y - coefficients[1+m*2]*sumOf_x)/n;
	} else { // The given coefficient values that were obtained for the Kernel are the ones obtained from the multiple polynomial regression that was applied during the training of its model.
		// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (coefficients).
		int N = 2; // This variable is used to store the order of degree of the machine learning model that was trained.
		int mTimesNPlusOne = m*N+1; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentRowTimesmTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
		int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		float increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
		int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentColumnMinusOne; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentColumnMinusOneTimesN; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		for (int currentRow=0; currentRow<n; currentRow++) {
			// -------------------- APLICATION OF THE KERNEL FUNCTION -------------------- //
			currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
			currentRowTimesM = currentRow*m;
			Y_tilde[currentRow] = coefficients[0];
			for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
				currentColumnMinusOne = currentColumn-1;
				currentColumnMinusOneTimesN = currentColumnMinusOne*N;
				currentRowAndColumn = currentColumnMinusOneTimesN + currentRowTimesmTimesNplusOne;
				increaseExponentialOfThisValue = 1;
				for (int currentExponential=1; currentExponential<(N+1); currentExponential++) {
					currentRowAndColumn2 = currentExponential + currentRowAndColumn;
					increaseExponentialOfThisValue = increaseExponentialOfThisValue * X[currentColumnMinusOne + currentRowTimesM];
					Y_tilde[currentRow] = Y_tilde[currentRow] + coefficients[currentExponential + currentColumnMinusOneTimesN]*increaseExponentialOfThisValue;
				}
			}
			Y_tilde[currentRow] = exp(Y_tilde[currentRow]); // We will store the results of the transformed function "K(x)" in "Y_tilde".
			
			// -------------------- SOLUTION OF "alpha" AND "b_0" -------------------- //
			// In order to obtain the coefficients of "alpha" and "b_0", we apply a linear regression using the input data "Y_tilde" (that was obtained with the transformed function "K(x)") with respect to the output data "Y_tilde".
			sumOf_xy += Y_tilde[currentRow] * Y[currentRow];
			sumOf_y += Y[currentRow];
			sumOf_x += Y_tilde[currentRow];
			sumOf_xSquared += Y_tilde[currentRow] * Y_tilde[currentRow];
		}
		
		// We calculate the value of "alpha" from the main function to be solved, which is "y_hat = alpha.K(x) + b_0"
		coefficients[2+2*m] = (n*sumOf_xy - sumOf_y*sumOf_x)/(n*sumOf_xSquared - sumOf_x*sumOf_x);
		
		// We calculate the value of "b_0" from the main function to be solved, which is "y_hat = alpha.K(x) + b_0"
		coefficients[1+2*m] = (sumOf_y - coefficients[2+2*m]*sumOf_x)/n;
	}
	
	// Free the Heap memory used for the locally allocated variables since they will no longer be used.
	free(Y_tilde);
	free(X_tilde);
	free(TransposeOf_X_tilde);
	free(matMul1);
	free(matMul2);
	return;
}


/**
* The "predictKernelMachineClassification()" function is used to
* make the predictions of the requested input values (X) by
* applying the Kernel machine classification model with the
* specified coefficient values. The predicted values will be
* stored in the argument pointer variable "Y_hat".
* 
* THIS IS IMPORTANT TO KNOW --> Finally, have in mind that not
* all the argument variables of this function will be used and
* the ones that do will strictly depend on what Kernel you
* tell this function to use. Therefore, make sure to read the
* following descriptions made for all the possible argument
* variables to be taken into consideration when using this
* function. There, it will be explicitly specified whether such
* argument variable will be used and under what cirumstances
* and what to do when it will not be used.
* 
* @param float *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'float' MEMORY
*					 SPACES.
*
* @param int N - This argument will only used by this function whenever
*				 the "polynomial" Kernel is selected with the argument
*				 variable "Kernel". For the case in which a different
*				 Kernel is chosen, then it is suggested to assign a
*				 value of "(int) 0" to "N". Conversely, if the
*				 "polynomial" Kernel is selected, then this argument
*				 variable will represent the order of degree that the
*				 machine learning model has.
*
* @param char Kernel[] - This argument will contain the string or array
*						 of characters that will specify the Kernel
*						 type requested by the implementer. Its
*						 possible values are the following:
*						 1) "linear" = applies the linear kernel.
*						 2) "polynomial" = applies the polynomial
*										   kernel.
*						 3) "logistic" = applies the logistic kernel.
*						 4) "gaussian" = applies the gaussian kernel.
*
* @param char isInteractionTerms = This argument variable is only used
*								   by this function whenever the
*								   "polynomial" Kernel is selected with
*								   the argument variable "Kernel". For
*								   the case in which a different Kernel
*								   is chosen, then assign a value of
*								   (int) 0 to this argument variable.
*								   Conversely, if you select the
*								   "polynomial" Kernel, then this
*								   argument variable will work as a flag
*								   to indicate whether the interaction
*								   terms are desired in the resulting
*								   model to be generated or not.
*						  		   Moreover, the possible valid values
*								   for this argument variable are:
*						  		   1) (int) 1 = The resulting model will
*												include the interaction
*												terms that are possible.
*												NOTE: This functionality,
*												is yet to be developed.
*						  		   2) (int) 0 = The resulting model will
*												not include interaction
*												terms.
*
* @param char isForceGaussianCurve = This argument variable will only be
*							used if the "gaussian" Kernel is chosen. For
*							the case in which a different Kernel is
*							selected, then it is suggested to define a
*						    value of "(int) 0" to this argument variable.
*							Conversely, if the "gaussian" Kernel is
*							chosen, then "isForceGaussianCurve" will
*							determine the types of coefficients to be
*							obtained during the training process. If
*							"isForceGaussianCurve" = 0, then the
*							coefficients to be obtained will be
*							directly from a model that has an
*							exponential with the multiple polynomial
*							equation without its interaction terms from
*							the following mathematical form "y_hat =
*							exp( b_tilde_{0} + b_tilde_{1}*x_{1} +
*							b_tilde_{2}*x_{1}^2 + b_tilde_{3}*x_{2} +
* 							b_tilde_{4}*x_{2}^2 + ... +
*							b_tilde_{2*m-1}*x_{m} +
* 							b_tilde_{2*m}*x_{m}^2 )".
*							If "isForceGaussianCurve" = 1, then the
*							coefficients to be obtained will be in the
*							from of an approximation from the previous
*							equation, such that each machine learning
*							feature will have a particular "mean" and
*							"variance" coefficient values. These will be
*							used to govern the following model instead
*							"y_hat =
*							exp(-( ((x_{1}-mean_{1})^2)/2*variance_{1} +
* 							((x_{2}-mean_{2})^2)/2*variance_{2} + ... +
* 							((x_{m}-mean_{m})^2)/2*variance_{m} ))".
*							On the other hand, have in consideration
*							that because the solution of the multiple
*							polynomial equation can sometimes give a non
*							perfect square binomial, when
*							"isForceGaussianCurve" = 0, you may not get
*							a true gaussian form under such circumstance.
*							Therefore, by setting "isForceGaussianCurve"
*							= 1, you will force the solution to always
*							have a gaussian form but this may increase
*							the total obtained error as a consequence.
*
* @param float *coefficients - This argument will contain the pointer
*					 to a memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. Moreover, because these
*					 coefficients may be stored in a different manner,
*					 depending on the Kernel that was chosen for the
*					 training process. The details of how the data is
*					 stored in "coefficients" and how to interpret them
*					 will be described in the particular static
*					 function that corresponds to the chosen Kernel.
*					 Due to this, the following will list the static
*					 function that will contain such information for
*					 each possible Kernel so that the implementer
*					 reads their commented documentation to learn
*					 those details.
*					 1) Kernel="linear"  -->  trainLinearKernel()
*					 2) Kernel="polynomial"  -->  trainPolynomialKernel()
*					 3) Kernel="logistic"  -->  trainLogisticKernel()
*					 4) Kernel="gaussian"  -->  trainGaussianKernel()
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param float *Y_hat - This argument will contain the pointer to a
*					 	 memory allocated output matrix, representing
*					 	 the predicted data of the system under study.
*						 THIS VARIABLE SHOULD BE ALLOCATED BEFORE
*						 CALLING THIS FUNCTION WITH A SIZE OF "n"
*						 TIMES "p=1" 'float' MEMORY SPACES. The
*						 results will be stored in the same order as
*						 the input data given such that the first
*						 sample will be stored in the row with index
*						 "0" and the last sample in the row with
*						 index "n". Finally, the possible output
*						 values are either "+1" or "-1".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 27, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void predictKernelMachineClassification(float *X, int N, char Kernel[], char isInteractionTerms, char isForceGaussianCurve, float *coefficients, int n, int m, int p, float *Y_hat) {
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m < 1) {
		Serial.print("\nERROR: The machine learning features (independent variables) must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		Serial.print("\nERROR: With respect to the system under study, there must only be only one output for this particular algorithm.\n");
		exit(1);
	}
	
	// We predict all the requested input values (X) with the desired machine learning algorithm; Kernel and; coefficient values ("Beta_0", "Beta_1", ..., "Beta_m", "b_0" and "alpha").
	// ------------------------ LINEAR KERNEL ------------------------ //
	if (strcmp(Kernel, "linear") == 0) {
		// NOTE: For performance purposes, the coefficient values of "alpha" and "b_0" from the main equation "y_hat = alpha.K(x) + b_0" will be ignored for this particular Kernel since they will have no effect on the result.
		int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
		int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		for (int currentRow=0; currentRow<n; currentRow++) {
			Y_hat[currentRow] = coefficients[0];
			currentRowTimesM = currentRow*m;
			for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
				Y_hat[currentRow] = Y_hat[currentRow] + coefficients[currentColumn]*X[currentColumn-1 + currentRowTimesM];
			}
			if (Y_hat[currentRow] > 0) { // We determine whether the model predicts the result of "+1" or "-1".
				Y_hat[currentRow] = 1;
			} else {
				Y_hat[currentRow] = -1;
			}
		}
	
	// ---------------------- POLYNOMIAL KERNEL ---------------------- //	
	} else if (strcmp(Kernel, "polynomial") == 0) {
		// Determine whether the interaction terms are available in the model to be used or not and then excecute the corresponding code.
		if (isInteractionTerms == 1) { // The interaction terms are available in the current model.
			Serial.print("\nERROR: The functionality of this function, when the argument variable \"isInteractionTerms\" contains a value of 1, has not yet been developed.\n");
			exit(1);
		} else if (isInteractionTerms == 0) { // The interaction terms are not available in the current model.			
			// NOTE: For performance purposes, the coefficient values of "alpha" and "b_0" from the main equation "y_hat = alpha.K(x) + b_0" will be ignored for this particular Kernel since they will have no effect on the result.
			// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (coefficients).
			int mTimesNPlusOne = m*N+1; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			int currentRowTimesmTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
			int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
			int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
			int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			float increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
			int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			int currentColumnMinusOne; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			int currentColumnMinusOneTimesN; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			for (int currentRow=0; currentRow<n; currentRow++) {
				currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
				currentRowTimesM = currentRow*m;
				Y_hat[currentRow] = coefficients[0];
				for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
					currentColumnMinusOne = currentColumn-1;
					currentColumnMinusOneTimesN = currentColumnMinusOne*N;
					currentRowAndColumn = currentColumnMinusOneTimesN + currentRowTimesmTimesNplusOne;
					increaseExponentialOfThisValue = 1;
					for (int currentExponential=1; currentExponential<(N+1); currentExponential++) {
						currentRowAndColumn2 = currentExponential + currentRowAndColumn;
						increaseExponentialOfThisValue = increaseExponentialOfThisValue * X[currentColumnMinusOne + currentRowTimesM];
						Y_hat[currentRow] = Y_hat[currentRow] + coefficients[currentExponential + currentColumnMinusOneTimesN]*increaseExponentialOfThisValue;
					}
				}
				if (Y_hat[currentRow] > 0) { // We determine whether the model predicts the result of "+1" or "-1".
					Y_hat[currentRow] = 1;
				} else {
					Y_hat[currentRow] = -1;
				}
			}
		} else { // The argument variable "isInteractionTerms" has been assigned an invalid value. Therefore, inform the user about this and terminate the program.
			Serial.print("\nERROR: The argument variable \"isInteractionTerms\" is meant to store only a binary value that equals either 1 or 0.\n");
			exit(1);
		}
	
	// ----------------------- LOGISTIC KERNEL ----------------------- //
	} else if (strcmp(Kernel, "logistic") == 0) {
		// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (coefficients).
		int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
		int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		for (int currentRow=0; currentRow<n; currentRow++) {
			Y_hat[currentRow] = coefficients[0];
			currentRowTimesM = currentRow*m;
			for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
				Y_hat[currentRow] = Y_hat[currentRow] + coefficients[currentColumn]*X[currentColumn-1 + currentRowTimesM];
			}
			Y_hat[currentRow] = 1 / (1 + exp(-Y_hat[currentRow])); // We finish applying the Kernel function "K(x)".
			Y_hat[currentRow] = coefficients[1+m] + coefficients[2+m]*Y_hat[currentRow]; // We input the value obtained with the Kernel function "K(x)" into the main equation of the Kernel machine classifer "y_hat = alpha.K(x) + b_0".
			if (Y_hat[currentRow] > 0) { // We determine whether the model predicts the result of "+1" or "-1".
				Y_hat[currentRow] = 1;
			} else {
				Y_hat[currentRow] = -1;
			}
		}
	
	// ----------------------- GAUSSIAN KERNEL ----------------------- //
	} else if (strcmp(Kernel, "gaussian") == 0) {
		// Determine the types of coefficient values that were given for the model that was trained under the gaussian regression.
		if (isForceGaussianCurve == 1) { // The given coefficient values are the literal mean and variance.
			// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (coefficients).
			float squareThisValue; // Variable used to store the value that wants to be squared.
			int currentColumnTimesTwo; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			for (int currentRow=0; currentRow<n; currentRow++) {
				Y_hat[currentRow] = 0;
				for (int currentColumn=0; currentColumn<m; currentColumn++) {
					currentColumnTimesTwo = currentColumn*2;
					squareThisValue = X[currentColumn + currentRow*m] - coefficients[currentColumnTimesTwo];
					Y_hat[currentRow] = Y_hat[currentRow] + squareThisValue * squareThisValue / (2*coefficients[1 + currentColumnTimesTwo]);
				}
				Y_hat[currentRow] = exp(-Y_hat[currentRow]); // We finish applying the Kernel function "K(x)".
				Y_hat[currentRow] = coefficients[m*2] + coefficients[1+m*2]*Y_hat[currentRow]; // We input the value obtained with the Kernel function "K(x)" into the main equation of the Kernel machine classifer "y_hat = alpha.K(x) + b_0".
				if (Y_hat[currentRow] > 0) { // We determine whether the model predicts the result of "+1" or "-1".
					Y_hat[currentRow] = 1;
				} else {
					Y_hat[currentRow] = -1;
				}
			}
		} else if (isForceGaussianCurve == 0) { // The given coefficient values are the ones obtained from the multiple polynomial regression that was applied during the training of this model.
			// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (coefficients).
			int N = 2; // This variable is used to store the order of degree of the machine learning model that was trained.
			int mTimesNPlusOne = m*N+1; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			int currentRowTimesmTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
			int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
			int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
			int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			float increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
			int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			int currentColumnMinusOne; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			int currentColumnMinusOneTimesN; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
			for (int currentRow=0; currentRow<n; currentRow++) {
				currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
				currentRowTimesM = currentRow*m;
				Y_hat[currentRow] = coefficients[0];
				for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
					currentColumnMinusOne = currentColumn-1;
					currentColumnMinusOneTimesN = currentColumnMinusOne*N;
					currentRowAndColumn = currentColumnMinusOneTimesN + currentRowTimesmTimesNplusOne;
					increaseExponentialOfThisValue = 1;
					for (int currentExponential=1; currentExponential<(N+1); currentExponential++) {
						currentRowAndColumn2 = currentExponential + currentRowAndColumn;
						increaseExponentialOfThisValue = increaseExponentialOfThisValue * X[currentColumnMinusOne + currentRowTimesM];
						Y_hat[currentRow] = Y_hat[currentRow] + coefficients[currentExponential + currentColumnMinusOneTimesN]*increaseExponentialOfThisValue;
					}
				}
				Y_hat[currentRow] = exp(Y_hat[currentRow]); // We finish applying the Kernel function "K(x)".
				Y_hat[currentRow] = coefficients[1+2*m] + coefficients[2+2*m]*Y_hat[currentRow]; // We input the value obtained with the Kernel function "K(x)" into the main equation of the Kernel machine classifer "y_hat = alpha.K(x) + b_0".
				if (Y_hat[currentRow] > 0) { // We determine whether the model predicts the result of "+1" or "-1".
					Y_hat[currentRow] = 1;
				} else {
					Y_hat[currentRow] = -1;
				}
			}
		} else { // The argument variable "isForceGaussianCurve" has been assigned an invalid value. Therefore, inform the user about this and terminate the program.
			Serial.print("\nERROR: Please assign a valid value for the flag \"isForceGaussianCurve\", which may be 1 or 0.\n");
			exit(1);
		}
	
	// ------------------- INVALID REQUESTED KERNEL ------------------ //
	} else {
		Serial.print("\nERROR: The requested Kernel has not yet been implemented in the CenyML library. Please assign a valid one.\n");
		exit(1);
	}
	
	return;
}




#endif

