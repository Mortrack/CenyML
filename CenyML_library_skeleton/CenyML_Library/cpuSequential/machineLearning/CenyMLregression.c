/*
   Copyright 2022 Cesar Miranda Meza (alias: Mortrack)

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
#include "CenyMLregression.h"
// IMPORTANT NOTE: This library uses the math.h library and therefore,
//				   remember to use the "-lm" flag when compiling it.


/**
* The "getSimpleLinearRegression()" function is used to apply the
* machine learning algorithm called simple linear regression as
* formulated in the master thesis of Cesar Miranda Meza called
* "Machine learning to support applications with embedded systems
* and parallel computing". Within this process, the best fitting
* equation with the form of "y_hat = b_0 + b_1*x" will be
* identified with respect to the sampled data given through the
* argument pointer variables "X" and "Y". As a result, the
* identified coefficient values will be stored in the argument
* pointer variable "b".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m=1" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
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
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable in which we will store
*					 the identified best fitting coefficient values
*					 for the desired machine learning algorithm. These
*					 coefficients will each be stored in the same
*					 column but under different rows where the first
*					 coefficient (b_0) will be stored in the row with
*					 index 0 and the last coefficient (b_1) will be
*					 stored in the row with index 1. IT IS
*					 INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*					 BEFORE CALLING THIS FUNCTION WITH A VARIABLE SIZE
*					 OF "m+1=2" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 13, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getSimpleLinearRegression(double *X, double *Y, int n, int m, int p, double *b) {
	// If the machine learning features exceed the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m != 1) {
		printf("\nERROR: The machine learning features (independent variables) must be equal to 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study exceed the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		printf("\nERROR: The outputs of the system under study must be equal to 1 for this particular algorithm.\n");
		exit(1);
	}
	
	// -------------------- SOLUTION OF THE MODEL -------------------- //
	// We declare and innitialize the variables that will be required to calculate the coefficients of "b".
	double sumOf_xy = 0;
	double sumOf_y = 0;
	double sumOf_x = 0;
	double sumOf_xSquared = 0;
	
	// We calculate the necessary summations in order to obtain the coefficients of "b".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		sumOf_xy += X[currentRow] * Y[currentRow];
		sumOf_y += Y[currentRow];
		sumOf_x += X[currentRow];
		sumOf_xSquared += X[currentRow] * X[currentRow];
	}
	
	// We calculate the value of the coefficient "b_1".
	b[1] = (n*sumOf_xy - sumOf_y*sumOf_x)/(n*sumOf_xSquared - sumOf_x*sumOf_x);
	
	// We calculate the value of the coefficient "b_0".
	b[0] = (sumOf_y - b[1]*sumOf_x)/n;
	
	return;
}


/**
* The "predictSimpleLinearRegression()" function is used to make the
* predictions of the requested input values (X) by applying the
* simple linear equation system with the specified coefficient values
* (b). The predicted values will be stored in the argument pointer
* variable "Y_hat".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning predictions will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m=1" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE IS ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A VARIABLE
*					 SIZE OF "m+1=2" 'DOUBLE' MEMORY SPACES.
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
*						 TIMES "p=1" 'DOUBLE' MEMORY SPACES. The
*						 results will be stored in the same order as
*						 the input data given such that the first
*						 sample will be stored in the row with index
*						 "0" and the last sample in the row with
*						 index "n".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 13, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void predictSimpleLinearRegression(double *X, double *b, int n, int m, int p, double *Y_hat) {
	// If the machine learning features exceed the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m != 1) {
		printf("\nERROR: The machine learning features (independent variables) must be equal to 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study exceed the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		printf("\nERROR: The outputs of the system under study must be equal to 1 for this particular algorithm.\n");
		exit(1);
	}
	
	// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (b).
	for (int currentRow = 0; currentRow < n; currentRow++) {
		Y_hat[currentRow] = b[0] + b[1]*X[currentRow];
	}
	
	return;
}


/**
* The "getMultipleLinearRegression()" function is used to apply the
* machine learning algorithm called multiple linear regression as
* formulated in the master thesis of Cesar Miranda Meza called
* "Machine learning to support applications with embedded systems
* and parallel computing". Within this process, the best fitting
* equation with the form of "y_hat = b_0 + b_1*x_1 + b_2*x_2
* + ... +  + b_m*x_m" will be identified with respect to the
* sampled data given through the argument pointer variables "X" and
* "Y". As a result, the identified coefficient values will be stored
* in the argument pointer variable "b".
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
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
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
*					 AND INITIALIZED WITH ZEROS BEFORE CALLING THIS
*					 FUNCTION WITH A VARIABLE SIZE OF "m+1" TIMES "p=1"
*					 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 17, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getMultipleLinearRegression(double *X, double *Y, int n, int m, int p, char isVariableOptimizer, double *b) {
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
* The "predictMultipleLinearRegression()" function is used to make the
* predictions of the requested input values (X) by applying the
* multiple linear equation system with the specified coefficient values
* (b). The predicted values will be stored in the argument pointer
* variable "Y_hat".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning predictions will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE IS ALLOCATED AND INITIALIZED
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
*						 TIMES "p=1" 'DOUBLE' MEMORY SPACES. The
*						 results will be stored in the same order as
*						 the input data given such that the first
*						 sample will be stored in the row with index
*						 "0" and the last sample in the row with
*						 index "n".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 17, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void predictMultipleLinearRegression(double *X, double *b, int n, int m, int p, double *Y_hat) {
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
	
	// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (b).
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow=0; currentRow<n; currentRow++) {
		Y_hat[currentRow] = b[0];
		currentRowTimesM = currentRow*m;
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			Y_hat[currentRow] = Y_hat[currentRow] + b[currentColumn]*X[currentColumn-1 + currentRowTimesM];
		}
	}
	
	return;
}


/**
* The "getPolynomialRegression()" function is used to apply the
* machine learning algorithm called polynomial regression as
* formulated in the master thesis of Cesar Miranda Meza called
* "Machine learning to support applications with embedded systems
* and parallel computing". Within this process, the best fitting
* equation with the form of "y_hat = b_0 + b_1*x + b_2*x^2 +
* b_3*x^3 + ... +  + b_N*x^N" will be identified with respect to
* the sampled data given through the argument pointer variables
* "X" and "Y". As a result, the identified coefficient values
* will be stored in the argument pointer variable "b".
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
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m=1" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
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
* @param int N - This argument will represent the desired order of
*				 degree for the machine learning model to be trained.
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
*					 index 0 and the last coefficient (b_N) will be
*					 stored in the row with index "N". IT IS
*					 INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*					 AND INITIALIZED WITH ZEROS BEFORE CALLING THIS
*					 FUNCTION WITH A VARIABLE SIZE OF "N+1" TIMES "p=1"
*					 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 18, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getPolynomialRegression(double *X, double *Y, int n, int m, int p, int N, char isVariableOptimizer, double *b) {
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m != 1) {
		printf("\nERROR: The machine learning features (independent variables) must be equal to 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		printf("\nERROR: With respect to the system under study, there must only be only one output for this particular algorithm.\n");
		exit(1);
	}
	
	// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
	// Store the data that must be contained in the input matrix "X_tilde". In addition, we obtain the transpose of "X_tilde".
	int currentRowTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int NplusOne = N+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	double *X_tilde = (double *) malloc(n*NplusOne*sizeof(double)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
	double *TransposeOf_X_tilde = (double *) malloc(NplusOne*n*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
	double increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
	for (int currentRow=0; currentRow<n; currentRow++) {
		currentRow2 = 0; // We reset the counters used in the following for-loop.
		currentRowTimesNplusOne = currentRow*NplusOne;
		X_tilde[currentRowTimesNplusOne] = 1;
		TransposeOf_X_tilde[currentColumn2] = X_tilde[currentRowTimesNplusOne];
		increaseExponentialOfThisValue = 1;
		for (int currentExponential=1; currentExponential<NplusOne; currentExponential++) {
			currentRowAndColumn = currentExponential + currentRowTimesNplusOne;
			increaseExponentialOfThisValue = increaseExponentialOfThisValue * X[currentRow];
			X_tilde[currentRowAndColumn] = increaseExponentialOfThisValue;
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
	double *matMul1 = (double *) calloc(NplusOne*NplusOne, sizeof(double)); // We allocate, and initialize with zeros, the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "X_tilde" and its transpose.
	for (int currentRow=0; currentRow<NplusOne; currentRow++) {
		currentRowTimesNplusOne = currentRow*NplusOne;
		currentRowTimesN = currentRow*n;
		for (int currentColumn=0; currentColumn<NplusOne; currentColumn++) {
			currentColumnTimesN = currentColumn*n;
			currentRowAndColumn = currentColumn + currentRowTimesNplusOne;
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
	for (int currentRow=0; currentRow<NplusOne; currentRow++) { // We set all values to zero.
		currentRowTimesNplusOne = currentRow*NplusOne;
		for (int currentColumn=0; currentColumn<NplusOne; currentColumn++) {
			TransposeOf_X_tilde[currentColumn + currentRowTimesNplusOne] = 0;
	    }
	}
	for (int currentUnitaryValue=0; currentUnitaryValue<NplusOne; currentUnitaryValue++) { // We set the corresponding 1's values to make the corresponding unitary matrix.
		TransposeOf_X_tilde[currentUnitaryValue + currentUnitaryValue*NplusOne] = 1;
	}
	
	// In order to continue obtaining the coefficients, we calculate the matrix inverse of "matMul1" with the Gauss-Jordan approach and store its result in "TransposeOf_X_tilde".
	int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	double ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
	for (int currentColumn=0; currentColumn<NplusOne; currentColumn++) { // We apply the differentiations applied to each row according to the approach used.
		currentColumnTimesMplusOne = currentColumn*NplusOne;
		currentRowAndColumn2 = currentColumn + currentColumnTimesMplusOne;
		for (int currentRow=0; currentRow<NplusOne; currentRow++) {
			if (currentRow != currentColumn) {
				currentRowTimesNplusOne = currentRow*NplusOne;
				ratioModifier = matMul1[currentColumn + currentRowTimesNplusOne]/matMul1[currentRowAndColumn2];
				for (int currentModifiedElements=0; currentModifiedElements<NplusOne; currentModifiedElements++) { // We apply the current process to the principal matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesNplusOne;
					matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] - ratioModifier * matMul1[currentModifiedElements + currentColumnTimesMplusOne];
				}
				for (int currentModifiedElements=0; currentModifiedElements<NplusOne; currentModifiedElements++) { // We apply the current process to the result matrix.
					currentRowAndColumn = currentModifiedElements + currentRowTimesNplusOne;
					TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] - ratioModifier * TransposeOf_X_tilde[currentModifiedElements + currentColumnTimesMplusOne];
				}
			}
		}
    }
	for (int currentRow=0; currentRow<NplusOne; currentRow++) { // We apply the last step of the approach used in order to obtain the diagonal of 1's in the principal matrix.
		currentRowTimesNplusOne = currentRow*NplusOne;
		currentRowAndColumn2 = currentRow + currentRowTimesNplusOne;
		for (int currentColumn=0; currentColumn<NplusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesNplusOne;
			TransposeOf_X_tilde[currentRowAndColumn] = TransposeOf_X_tilde[currentRowAndColumn] / matMul1[currentRowAndColumn2];
		}
    }
    
	// In order to continue obtaining the coefficients, we multiply the inverse matrix that was obtained by the transpose of the matrix "X_tilde".
	// NOTE: Remember that we will get the data of the transpose of
	//		 "X_tilde" directly from that same variable
	//		 ("X_tilde") due to performance reasons and; that the
	//		 inverse matrix that was obtained is stored in
	//		 "TransposeOf_X_tilde".
	double *matMul2 = (double *) calloc(NplusOne*n, sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "X_tilde".
	for (int currentRow=0; currentRow<NplusOne; currentRow++) {
		currentRowTimesN = currentRow*n;
		currentRowTimesNplusOne = currentRow*NplusOne;
		for (int currentColumn=0; currentColumn<n; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesN;
			currentColumnTimesMplusOne = currentColumn*NplusOne;
			for (int currentMultipliedElements=0; currentMultipliedElements<NplusOne; currentMultipliedElements++) {
				matMul2[currentRowAndColumn] = matMul2[currentRowAndColumn] + TransposeOf_X_tilde[currentMultipliedElements + currentRowTimesNplusOne] * X_tilde[currentMultipliedElements + currentColumnTimesMplusOne];
			}
	    }
	}
	
	// In order to conclude obtaining the coefficients ("b"), we multiply the previously resulting matrix ("matMul2") by the output matrix "Y".
	for (int currentRow=0; currentRow<NplusOne; currentRow++) {
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
* The "predictMultipleLinearRegression()" function is used to make the
* predictions of the requested input values (X) by applying the
* polynomial equation system with the specified order of degree (N)
* and coefficient values (b). The predicted values will be stored in
* the argument pointer variable "Y_hat".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning predictions will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m=1" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param int N - This argument will represent the desired order of
*				 degree of the machine learning model to be used.
*
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE IS ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A VARIABLE
*					 SIZE OF "N+1" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
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
*						 TIMES "p=1" 'DOUBLE' MEMORY SPACES. The
*						 results will be stored in the same order as
*						 the input data given such that the first
*						 sample will be stored in the row with index
*						 "0" and the last sample in the row with
*						 index "n".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 18, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void predictPolynomialRegression(double *X, int N, double *b, int n, int m, int p, double *Y_hat) {
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m != 1) {
		printf("\nERROR: The machine learning features (independent variables) must be equal to 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		printf("\nERROR: With respect to the system under study, there must only be only one output for this particular algorithm.\n");
		exit(1);
	}
	
	// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (b).
	int NplusOne = N+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	double increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
	for (int currentRow=0; currentRow<n; currentRow++) {
		Y_hat[currentRow] = b[0];
		increaseExponentialOfThisValue = 1;
		for (int currentExponential=1; currentExponential<NplusOne; currentExponential++) {
			increaseExponentialOfThisValue = increaseExponentialOfThisValue * X[currentRow];
			Y_hat[currentRow] = Y_hat[currentRow] + b[currentExponential]*increaseExponentialOfThisValue;
		}
	}
	
	return;
}


/**
* The "getMultiplePolynomialRegression()" function is used to apply
* the machine learning algorithm called multiple polynomial
* regression as formulated in the master thesis of Cesar Miranda
* Meza called "Machine learning to support applications with
* embedded systems and parallel computing". Within this process,
* for the case of not having included the interaction terms, the
* best fitting equation with the form of "y_hat = b_{0}
* + b_{1}*x_{1} + b_{2}*x_{1}^{2} + ... + b_{N}*x_{1}^{N}
* + b_{N+1}*x_{2} + b_{N+2}*x_{2}^{2} + ... + b_{2*N}*x_{2}^{N}
* + ... + b_{2*N+1}*x_{m} + b_{2*N+2}*x_{m}^{2}
* + ... + b_{m*N}*x_{m}^{N}" will be identified with respect to the
* sampled data given through the argument pointer variables "X" and
* "Y". As a result, the identified coefficient values will be stored
* in the argument pointer variable "b".
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
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
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
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable in which we will store
*					 the identified best fitting coefficient values
*					 for the desired machine learning algorithm. These
*					 coefficients will each be stored in the same
*					 column but under different rows where the first
*					 coefficient (b_{0}) will be stored in the row with
*					 index 0 and the last coefficient (b_{m*N}) will be
*					 stored in the row with index "m*N". IT IS
*					 INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*					 AND INITIALIZED WITH ZEROS BEFORE CALLING THIS
*					 FUNCTION WITH A VARIABLE SIZE OF "m*N+1" TIMES "p=1"
*					 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 18, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getMultiplePolynomialRegression(double *X, double *Y, int n, int m, int p, int N, char isInteractionTerms, char isVariableOptimizer, double *b) {
	// Determine whether the interaction terms are desired in the resulting model to be generated or not and then excecute the corresponding code.
	if (isInteractionTerms == 1) { // Include the interaction terms in the training process of the model to be generated.
		printf("\nERROR: The functionality of this function, when the argument variable \"isInteractionTerms\" contains a value of 1, has not yet been developed.\n");
		exit(1);
		
		
	} else if (isInteractionTerms == 0) { // Do not inlcude the interaction terms in the training process of the model to be generated.
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
		
		// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
		// Store the data that must be contained in the input matrix "X_tilde". In addition, we obtain the transpose of "X_tilde".
		int currentRowTimesmTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int mTimesNPlusOne = m*N+1; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
		int NplusOne = (N+1); //This variable is used to store a repetitive matheamtical operation, for performance purposes.
		double *X_tilde = (double *) malloc(n*mTimesNPlusOne*sizeof(double)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
		double *TransposeOf_X_tilde = (double *) malloc(mTimesNPlusOne*n*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
		int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
		int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
		double increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
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
		// In order to start obtaining the coefficients, we multiply the matrix "TransposeOf_X_tilde" with the matrix "X_tilde" and store the result in the matrix "matMul1".
		int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int currentColumnTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		double *matMul1 = (double *) calloc(mTimesNPlusOne*mTimesNPlusOne, sizeof(double)); // We allocate, and initialize with zeros, the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "X_tilde" and its transpose.
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
		double ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
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
		double *matMul2 = (double *) calloc(mTimesNPlusOne*n, sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "X_tilde".
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
		
		// In order to conclude obtaining the coefficients ("b"), we multiply the previously resulting matrix ("matMul2") by the output matrix "Y".
		for (int currentRow=0; currentRow<mTimesNPlusOne; currentRow++) {
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
		
		
	} else { // The argument variable "isInteractionTerms" has been assigned an invalid value. Therefore, inform the user about this and terminate the program.
		printf("\nERROR: The argument variable \"isInteractionTerms\" is meant to store only a binary value that equals either 1 or 0.\n");
		exit(1);
	}
	
	return;
}


/**
* The "predictMultiplePolynomialRegression()" function is used to
* make the predictions of the requested input values (X) by applying
* the multiple polynomial equation system with the specified order
* of degree (N) and coefficient values (b). The predicted values
* will be stored in the argument pointer variable "Y_hat".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning predictions will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param int N - This argument will represent the desired order of
*				 degree of the machine learning model to be used.
*
* @param char isInteractionTerms = This argument variable will work as a
*						  		   flag to indicate whether the
*								   interaction terms are available in the
*								   model to be used or not.
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
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE IS ALLOCATED AND INITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A VARIABLE
*					 SIZE OF "m*N+1" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
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
*						 TIMES "p=1" 'DOUBLE' MEMORY SPACES. The
*						 results will be stored in the same order as
*						 the input data given such that the first
*						 sample will be stored in the row with index
*						 "0" and the last sample in the row with
*						 index "n".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 18, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void predictMultiplePolynomialRegression(double *X, int N, char isInteractionTerms, double *b, int n, int m, int p, double *Y_hat) {
	// Determine whether the interaction terms are available in the model to be used or not and then excecute the corresponding code.
	if (isInteractionTerms == 1) { // The interaction terms are available in the current model.
		printf("\nERROR: The functionality of this function, when the argument variable \"isInteractionTerms\" contains a value of 1, has not yet been developed.\n");
		exit(1);
		
		
	} else if (isInteractionTerms == 0) { // The interaction terms are not available in the current model.
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
		
		// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (b).
		int mTimesNPlusOne = m*N+1; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentRowTimesmTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
		int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		double increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
		int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentColumnMinusOne; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentColumnMinusOneTimesN; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		for (int currentRow=0; currentRow<n; currentRow++) {
			currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
			currentRowTimesM = currentRow*m;
			Y_hat[currentRow] = b[0];
			for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
				currentColumnMinusOne = currentColumn-1;
				currentColumnMinusOneTimesN = currentColumnMinusOne*N;
				currentRowAndColumn = currentColumnMinusOneTimesN + currentRowTimesmTimesNplusOne;
				increaseExponentialOfThisValue = 1;
				for (int currentExponential=1; currentExponential<(N+1); currentExponential++) {
					currentRowAndColumn2 = currentExponential + currentRowAndColumn;
					increaseExponentialOfThisValue = increaseExponentialOfThisValue * X[currentColumnMinusOne + currentRowTimesM];
					Y_hat[currentRow] = Y_hat[currentRow] + b[currentExponential + currentColumnMinusOneTimesN]*increaseExponentialOfThisValue;
				}
			}
		}
		
		
	} else { // The argument variable "isInteractionTerms" has been assigned an invalid value. Therefore, inform the user about this and terminate the program.
		printf("\nERROR: The argument variable \"isInteractionTerms\" is meant to store only a binary value that equals either 1 or 0.\n");
		exit(1);
	}
	
	return;
}


/**
* The "getLogisticRegression()" function is used to apply the
* machine learning algorithm called logistic regression as
* formulated in the master thesis of Cesar Miranda Meza called
* "Machine learning to support applications with embedded systems
* and parallel computing". Within this process, the best fitting
* equation with the form of "y_hat = 1 / (1+e^{-(b_{0}
* + b_{1}x_{1} + b_{2}x_{2} + ... + b_{m}x_{m})})" will be
* identified with respect to the sampled data given through the
* argument pointer variables "X" and "Y". As a result, the
* identified coefficient values will be stored in the argument
* pointer variable "b".
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
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
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
*					 AND INITIALIZED WITH ZEROS BEFORE CALLING THIS
*					 FUNCTION WITH A VARIABLE SIZE OF "m+1" TIMES "p=1"
*					 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 19, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getLogisticRegression(double *X, double *Y, int n, int m, int p, char isVariableOptimizer, double *b) {
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
	
	
	// Store the data that must be contained in the input matrix "X_tilde". In addition, we obtain the transpose of "X_tilde".
	int currentRowTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	double *Y_tilde = (double *) malloc(n*p*sizeof(double)); // This variable will contain the output data of the system under study ("Y") as required by the training of this algorithm.
	double *X_tilde = (double *) malloc(n*mPlusOne*sizeof(double)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
	double *TransposeOf_X_tilde = (double *) malloc(mPlusOne*n*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
	for (int currentRow=0; currentRow<n; currentRow++) {
		// --------------- PREPROCESSING OF THE OUTPUT DATA --------------- //
		if (Y[currentRow] <= 0) {
			printf("\nERROR: The output data from the row %d and column %d, had a value that is equal or less than zero. Please assign the proper output values for this algorithm, considering the restriction: 0 < y_{i,k} < 1.\n", currentRow, p);
			exit(1);
		}
		if (Y[currentRow] >= 1) {
			printf("\nERROR: The output data from the row %d and column %d, had a value that is equal or greater than one. Please assign the proper output values for this algorithm, considering the restriction: 0 < y_{i,k} < 1.\n", currentRow, p);
			exit(1);
		}
		Y_tilde[currentRow] = log(Y[currentRow]/(1-Y[currentRow]));
		
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
	return;
}


/**
* The "predictLogisticRegression()" function is used to make the
* predictions of the requested input values (X) by applying the
* logistic equation system with the specified coefficient values
* (b). The predicted values will be stored in the argument pointer
* variable "Y_hat".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning predictions will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE IS ALLOCATED AND INITIALIZED
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
*						 TIMES "p=1" 'DOUBLE' MEMORY SPACES. The
*						 results will be stored in the same order as
*						 the input data given such that the first
*						 sample will be stored in the row with index
*						 "0" and the last sample in the row with
*						 index "n".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 19, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void predictLogisticRegression(double *X, double *b, int n, int m, int p, double *Y_hat) {
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
	
	// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (b).
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow=0; currentRow<n; currentRow++) {
		Y_hat[currentRow] = b[0];
		currentRowTimesM = currentRow*m;
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			Y_hat[currentRow] = Y_hat[currentRow] + b[currentColumn]*X[currentColumn-1 + currentRowTimesM];
		}
		Y_hat[currentRow] = 1 / (1 + exp(-Y_hat[currentRow]));
	}
	
	return;
}


/**
* The "getGaussianRegression()" function is used to apply the
* machine learning algorithm called gaussian regression as
* formulated in the master thesis of Cesar Miranda Meza called
* "Machine learning to support applications with embedded systems
* and parallel computing". Within this process, the equation with
* the form of "y_hat = exp(-( ((x_{1}-mean_{1})^2)/2*variance_{1} +
* ((x_{2}-mean_{2})^2)/2*variance_{2} + ... +
* ((x_{m}-mean_{m})^2)/2*variance_{m} ))" will correspond to
* when the gaussian curve is forced. On the other hand, the
* equation with the form of "y_hat = exp( b_tilde_{0} +
* b_tilde_{1}*x_{1} + b_tilde_{2}*x_{1}^2 + b_tilde_{3}*x_{2} +
* b_tilde_{4}*x_{2}^2 + ... + b_tilde_{2*m-1}*x_{m} +
* b_tilde_{2*m}*x_{m}^2 )" will correspond to when the gaussian with
* multiple the polynomial equation without the interaction terms is
* used instead. For any of these two cases, their corresponding
* best fitting coefficient values will be identified with respect to
* the sampled data given through the argument pointer variables "X"
* and "Y". As a result, the identified coefficient values will be
* stored in the argument pointer variable "b".
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
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
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
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable in which we will store
*					 the identified best fitting coefficient values
*					 for the desired machine learning algorithm. IT IS
*					 INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*					 AND INITIALIZED WITH ZEROS BEFORE CALLING THIS
*					 FUNCTION WITH A VARIABLE SIZE OF "m*2+1" TIMES
*					 "p=1" 'DOUBLE' MEMORY SPACES. However, the
*					 coefficients will be stored in a different manner
*					 depending on the value of the argument variable
*					 "isForceGaussianCurve". If its value equals 0, the
*					 coefficients will each be stored in the same column
*					 but under different rows where the first coefficient
*					 (b_0) will be stored in the row with index 0 and the
*					 last coefficient (b_{2*m}) will be stored in the row
*					 with index "2*m". On the other hand, if
*					 "isForceGaussianCurve" = 1, then each machine learning
*					 feature will contain two coefficients. The first will
*					 be the "mean" and the second will be the "variance".
*					 Both of these will be stored in the same row, where
*					 the first column (column index 0) will store the mean
*					 and the variance will be stored in the second column
*					 (column index 1). In addition, each of the identified
*					 means and variances for each different machine learning
*					 feature will be stored separately in rows. The first
*					 row (row index 0) will contain the mean and variance
*					 for the first machine learning feature and the last row
*					 (row index "m") will contain the mean and variance for
*					 the "m"-th machine learning feature. Finally, note that
*					 if "isForceGaussianCurve" = 1, then the true size to be
*					 taken into account of the argument pointer variable "b"
*					 will be from "m*2+1" to "m*2". This means that the last
*					 index with respect to the entirely allocated memory in
*					 "b" will not be used or have any meaning under this
*					 case.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 21, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getGaussianRegression(double *X, double *Y, int n, int m, int p, char isForceGaussianCurve, char isVariableOptimizer, double *b) {
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
	// If the argument flag variable "isForceGaussianCurve" is different than the value of "1" and "0", then emit an error message and terminate the program. Otherwise, continue with the program.
	if (isForceGaussianCurve != 1) {
		if (isForceGaussianCurve != 0) {
			printf("\nERROR: Please assign a valid value for the flag \"isForceGaussianCurve\", which may be 1 or 0.\n");
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
	double *Y_tilde = (double *) malloc(n*p*sizeof(double)); // This variable will contain the output data of the system under study ("Y") as required by the training of this algorithm.
	double *X_tilde = (double *) malloc(n*mTimesNPlusOne*sizeof(double)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
	double *TransposeOf_X_tilde = (double *) malloc(mTimesNPlusOne*n*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
	double increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
	for (int currentRow=0; currentRow<n; currentRow++) {
		// --------------- PREPROCESSING OF THE OUTPUT DATA --------------- //
		if (Y[currentRow] <= 0) {
			printf("\nERROR: The output data from the row %d and column %d, had a value that is equal or less than zero. Please assign the proper output values for this algorithm, considering the restriction: 0 < y_{i,k} <= 1.\n", currentRow, p);
			exit(1);
		}
		if (Y[currentRow] > 1) {
			printf("\nERROR: The output data from the row %d and column %d, had a value that is greater than one. Please assign the proper output values for this algorithm, considering the restriction: 0 < y_{i,k} <= 1.\n", currentRow, p);
			exit(1);
		}
		Y_tilde[currentRow] = log(Y[currentRow]);
		
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
	
	// -------------------- SOLUTION OF THE MODEL -------------------- //
	// In order to start obtaining the coefficients, we multiply the matrix "TransposeOf_X_tilde" with the matrix "X_tilde" and store the result in the matrix "matMul1".
	int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentColumnTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	double *matMul1 = (double *) calloc(mTimesNPlusOne*mTimesNPlusOne, sizeof(double)); // We allocate, and initialize with zeros, the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "X_tilde" and its transpose.
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
	double ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
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
	double *matMul2 = (double *) calloc(mTimesNPlusOne*n, sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "X_tilde".
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
			b[currentRow] = b[currentRow] + matMul2[currentMultipliedElements + currentRowTimesN] * Y_tilde[currentMultipliedElements];
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
			b[currentRowTimesTwo] = b[onePlusCurrentRowTimesTwo];
			b[onePlusCurrentRowTimesTwo] = -1/(2*b[2 + currentRowTimesTwo]); // We obtain and store the approximation/expected value of the variance coefficient value of the current machine learning feature.
			b[currentRowTimesTwo] = b[currentRowTimesTwo] * b[onePlusCurrentRowTimesTwo]; // We obtain and store the approximation/expected value of the mean coefficient value of the current machine learning feature.
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
* The "predictGaussianRegression()" function is used to make the
* predictions of the requested input values (X) by applying the gaussian
* equation system with the specified coefficient values (b). The
* predicted values will be stored in the argument pointer variable
* "Y_hat".
*
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning predictions will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param char isForceGaussianCurve = This argument variable will
*							determine the types of coefficients that
*							were obtained during the training process.
*							If "isForceGaussianCurve" = 0, then the
*							coefficients were obtained directly from a
*							model that has an exponential with the
*							multiple polynomial equation without its
*							interaction terms from the following
*							mathematical form "y_hat = exp( b_tilde_{0}
*							+ b_tilde_{1}*x_{1} + b_tilde_{2}*x_{1}^2 +
*							b_tilde_{3}*x_{2} + b_tilde_{4}*x_{2}^2
*							+ ... + b_tilde_{2*m-1}*x_{m} +
* 							b_tilde_{2*m}*x_{m}^2 )".
*							If "isForceGaussianCurve" = 1, then the
*							coefficients were obtained as an
*							approximation from the previous equation,
*							such that each machine learning feature
*							will have a particular "mean" and "variance"
*							coefficient values. These will be used to
*							govern the following model instead "y_hat =
*							exp(-( ((x_{1}-mean_{1})^2)/2*variance_{1} +
* 							((x_{2}-mean_{2})^2)/2*variance_{2} + ... +
* 							((x_{m}-mean_{m})^2)/2*variance_{m} ))".
*
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE WAS ALLOCATED AND INITIALIZED
*					 BEFORE CALLING ITS TRAINING FUNCTION WITH A
*					 VARIABLE SIZE OF "m*2+1" TIMES "p=1" 'DOUBLE'
*					 MEMORY SPACES. This is because its actual used
*					 memory space will differ depending on the value
*					 defined on the argument pointer variable
*					 "isForceGaussianCurve" during the training
*					 process of the model. This is very important to
*					 have into consideration because when using the
*					 function "predictGaussianRegression()",
*					 "isForceGaussianCurve" should have the same value
*					 as in the training function in order to take into
*					 account the right coefficients and also the way
*					 they were distributed in "b".
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
*						 TIMES "p=1" 'DOUBLE' MEMORY SPACES. The
*						 results will be stored in the same order as
*						 the input data given such that the first
*						 sample will be stored in the row with index
*						 "0" and the last sample in the row with
*						 index "n".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
*
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 21, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void predictGaussianRegression(double *X, char isForceGaussianCurve, double *b, int n, int m, int p, double *Y_hat) {
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
	
	
	// Determine the types of coefficient values that were given for the model that was trained under the gaussian regression.
	if (isForceGaussianCurve == 1) { // The given coefficient values are the literal mean and variance.
		// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (b).
		double squareThisValue; // Variable used to store the value that wants to be squared.
		int currentColumnTimesTwo; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		for (int currentRow=0; currentRow<n; currentRow++) {
			Y_hat[currentRow] = 0;
			for (int currentColumn=0; currentColumn<m; currentColumn++) {
				currentColumnTimesTwo = currentColumn*2;
				squareThisValue = X[currentColumn + currentRow*m] - b[currentColumnTimesTwo];
				Y_hat[currentRow] = Y_hat[currentRow] + squareThisValue * squareThisValue / (2*b[1 + currentColumnTimesTwo]);
			}
			Y_hat[currentRow] = exp(-Y_hat[currentRow]);
		}
		
		
	} else if (isForceGaussianCurve == 0) { // The given coefficient values are the ones obtained from the multiple polynomial regression that was applied during the training of this model.
		// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (b).
		int N = 2; // This variable is used to store the order of degree of the machine learning model that was trained.
		int mTimesNPlusOne = m*N+1; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentRowTimesmTimesNplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
		int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		double increaseExponentialOfThisValue; // Variable used to store the value that wants to be raised exponentially.
		int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentColumnMinusOne; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		int currentColumnMinusOneTimesN; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
		for (int currentRow=0; currentRow<n; currentRow++) {
			currentRowTimesmTimesNplusOne = currentRow*mTimesNPlusOne;
			currentRowTimesM = currentRow*m;
			Y_hat[currentRow] = b[0];
			for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
				currentColumnMinusOne = currentColumn-1;
				currentColumnMinusOneTimesN = currentColumnMinusOne*N;
				currentRowAndColumn = currentColumnMinusOneTimesN + currentRowTimesmTimesNplusOne;
				increaseExponentialOfThisValue = 1;
				for (int currentExponential=1; currentExponential<(N+1); currentExponential++) {
					currentRowAndColumn2 = currentExponential + currentRowAndColumn;
					increaseExponentialOfThisValue = increaseExponentialOfThisValue * X[currentColumnMinusOne + currentRowTimesM];
					Y_hat[currentRow] = Y_hat[currentRow] + b[currentExponential + currentColumnMinusOneTimesN]*increaseExponentialOfThisValue;
				}
			}
			Y_hat[currentRow] = exp(Y_hat[currentRow]);
		}
		
		
	} else { // The argument variable "isForceGaussianCurve" has been assigned an invalid value. Therefore, inform the user about this and terminate the program.
		printf("\nERROR: Please assign a valid value for the flag \"isForceGaussianCurve\", which may be 1 or 0.\n");
		exit(1);
	}
	
	return;
}

