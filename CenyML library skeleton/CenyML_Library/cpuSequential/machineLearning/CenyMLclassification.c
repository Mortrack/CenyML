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
#include "CenyMLclassification.h"
// IMPORTANT NOTE: This library uses the math.h library and therefore,
//				   remember to use the "-lm" flag when compiling it.


/**
* The "getLinearLogisticClassification()" function is used to apply
* the machine learning algorithm called linear logistic
* classification. Within this process, the best fitting equation
* with the form of "y_hat = 1 / (1+e^{-(b_{0} + b_{1}x_{1} +
* b_{2}x_{2} + ... + b_{m}x_{m})})" will be identified with respect
* to the sampled data given through the argument pointer variables
* "X" and "Y". At the end of this algorithm, the identified
* coefficient values will be stored in the argument pointer variable
* "b". As a result, when inserting the coefficient values into this
* model, whenever its output is greater than the defined threshold
* (0 < threshold < 1), it should be interpreted as the model
* predicting that the current input values represent group 1 or the
* binary number "1". Conversely, if the model produces a value less
* than the defined threshold, it should be interpreted as the model
* predicting that the current input values represent group 2 or the
* binary number "0".
*
* NOTE: The algorithm section that applied the matrix inverse using
* the Gauss-Jordan method was inspired in the following source:
* "CodeSansar. Matrix Inverse Using Gauss Jordan Method C Program.
* November 16, 2021 (Recovery date), de CodeSansar Sitio web:
* https://bit.ly/3CowwSy".
*
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INNITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INNITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A SIZE OF
*					 "n" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
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
* @param double Y_epsilon -  This argument will contain the user defined
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
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable in which we will store
*					 the identified best fitting coefficient values
*					 for the desired machine learning algorithm. These
*					 coefficients will each be stored in the same
*					 column but under different rows where the first
*					 coefficient (b_0) will be stored in the row with
*					 index 0 and the last coefficient (b_m) will be
*					 stored in the row with index "m". IT IS
*					 INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*					 AND INNITIALIZED WITH ZEROS BEFORE CALLING THIS
*					 FUNCTION WITH A VARIABLE SIZE OF "m+1" TIMES "p=1"
*					 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: NOVEMBER 24, 2021
*/
void getLinearLogisticClassification(double *X, double *Y, int n, int m, int p, double Y_epsilon, char isVariableOptimizer, double *b) {
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m < 1) {
		printf("\nERROR: The machine learning features (independent variables) must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the samples are less than the number of machine learning features, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (n < m) {
		printf("\nERROR: The number of samples provided must be equal or higher than the number of machine learning features (independent variables) for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		printf("\nERROR: With respect to the system under study, there must only be only one output for this particular algorithm.\n");
		exit(1);
	}
	// If the output values of the system under study has a value that is not within the restrictions requested for this algorithm, then emit an error message and terminate the program.
	for (int currentRow=0; currentRow<n; currentRow++) {
		if (Y[currentRow] < 0) {
			printf("\nERROR: The output data from the row %d and column %d, had a value that is less than zero. Please assign the proper output values for this algorithm, considering the restriction: 0 <= y_{i,k} <= 1.\n", currentRow, p);
			exit(1);
		}
		if (Y[currentRow] > 1) {
			printf("\nERROR: The output data from the row %d and column %d, had a value that is greater than one. Please assign the proper output values for this algorithm, considering the restriction: 0 <= y_{i,k} <= 1.\n", currentRow, p);
			exit(1);
		}
	}
	
	// --------------- PREPROCESSING OF THE OUTPUT DATA --------------- //
	// Store the data that must be contained in the output matrix "Y_tilde".
	double *Y_tilde = (double *) malloc(n*p*sizeof(double)); // This variable will contain the output data of the system under study ("Y") as required by the training of this algorithm.
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
	double *X_tilde = (double *) malloc(n*mPlusOne*sizeof(double)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
	double *TransposeOf_X_tilde = (double *) malloc(mPlusOne*n*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
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
	double *matMul1 = (double *) calloc(mPlusOne*mPlusOne, sizeof(double)); // We allocate, and initialize with zeros, the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "X_tilde" and its transpose.
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
	double ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
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
	double *matMul2 = (double *) calloc(mPlusOne*n, sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "X_tilde".
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
}


/**
* The "predictLinearLogisticClassification()" function is used to
* make the predictions of the requested input values (X) by applying
* the linear logistic classification model with the specified
* coefficient values (b). The predicted values will be stored in the
* argument pointer variable "Y_hat".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning predictions will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INNITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double threshold - This argument will represent the value
*							that the implementer desires for the
*							threshold to be taken into account
*							during the predictions made by the
*							machine learning model that was trained.
*							Moreover, keep in mind the restriction
*							"0 < threshold < 1", which must be
*							complied.
*
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE IS ALLOCATED AND INNITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A VARIABLE
*					 SIZE OF "m+1" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
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
* @param double *Y_hat - This argument will contain the pointer to a
*					 	 memory allocated output matrix, representing
*					 	 the predicted data of the system under study.
*						 THIS VARIABLE SHOULD BE ALLOCATED BEFORE
*						 CALLING THIS FUNCTION WITH A SIZE OF "n"
*						 TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 24, 2021
* LAST UPDATE: N/A
*/
void predictLinearLogisticClassification(double *X, double threshold, double *b, int n, int m, int p, double *Y_hat) {
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m < 1) {
		printf("\nERROR: The machine learning features (independent variables) must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		printf("\nERROR: With respect to the system under study, there must only be only one output for this particular algorithm.\n");
		exit(1);
	}
	// If the specified threshold does not complied with the required restriction, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((threshold<=0) || (threshold>=1)) {
		printf("\nERROR: The specified threshold does not meet the restriction: 0 < threshold < 1.\n");
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
}

