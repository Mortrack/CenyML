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
#include "CenyMLregression.h"
// IMPORTANT NOTE: This library uses the math.h library and therefore,
//				   remember to use the "-lm" flag when compiling it.


/**
* The "getPermutations()" function is inspired in the code made and
* explained by "GeeksforGeeks" in the Youtube video "Write a program
* to print all permutations of a given string | GeeksforGeeks"
* (URL = https://www.youtube.com/watch?v=AfxHGNRtFac). Nonetheless,
* as coded in this file, this function is used to obtain all the
* possible permutations of the "m" elements contained in the first
* row of the pointer argument variable "inputMatrix". Finally, the
* results obtained will be stored in "inputMatrix" from its second
* row up to the !m-th ("!m" = factorial of m) row.
*
* @param int l - This argument will be used to determine one of
*				 the two values that will have to be swaped for
*				 the current permutation to be applied. When
*				 calling this function, it will be necessary to
*				 input this argument variable with the value of 0.
*
* @param int r - This argument will be used to determine one of
*				 the two values that will have to be swaped for
*				 the current permutation to be applied. When
*				 calling this function, it will be necessary to
*				 input this argument variable with the value of (m-1).
*
* @param int m - This argument will represent the total number
*				 of columns that the "inputMatrix" variable argument
*				 will have.
*
* @param int *currentRow - This argument will contain the pointer to
*						a memory allocated variable that will be used
*						to know in which row, of the argument pointer
*						variable "inputMatrix", to store the result of
*						the currently identified permutation. This is
*						necessary because this function will be solved
*						recursively and each time this function is
*						is called, a different permutation possition
*						will be obtained. NOTE THAT IT IS
*						INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*						BEFORE CALLING THIS FUNCTION AND INNITIALIZED
*						WITH A VALUE OF 0.
*
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   input matrix in which all the
*							   identified permutations with
*							   respect to the "m" elements of its first
*							   row, will be stored. These resulting
*							   permutation positions will be stored
*							   from the 2nd row up to the !m-th row
*							   ("!m" = factorial of m). NOTE THAT THIS
*							   VARIABLE MUST BE PREVIOUSLY ALLOCATED.
*							   FURTHERMORE, THE DATA TO BE PERMUTATED
*							   IS THE ONE THAT WAS STORED IN EACH
*							   COLUMN OF THE FIRST ROW OF THIS
*							   VARIABLE BEFORE CALLING THIS FUNCTION.
*							   FINNALLY, REMEMBER THAT THE TOTAL NUMBER
*							   OF PERMUTATIONS TO BE STORED WILL BE
*							   EQUAL TO "!m" AND THEREFORE, THIS
*							   VARIABLE WILL REQUIRE TO HAVE "m"
*							   COLUMNS AND "!m" ROWS ALLOCATED IN IT.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix". THERE, YOU WILL FIND ALL THE POSSIBLE
*		COMBINATION PERMUTATIONS WITH RESPECT TO THE DATA STORED IN THE
*		FIRST ROW OF "inputMatrix", EACH ONE IN A SEPERATED ROW.
*
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: OCTOBER 17, 2021
* LAST UPDATE: NOVEMBER 02, 2021
*/
static void getPermutations(int l, int r, int m, int *currentRow, int *inputMatrix) {
    int i;
    int dataHolder;
    if (l == r) {
    	// Store the currently identified permutation.
        for (int currentColumn=0; currentColumn<m; currentColumn++) {
        	inputMatrix[currentColumn + (*currentRow)*m] = inputMatrix[currentColumn];
		}
		
		// Indicate to the next recursive function to store the next permutation in the next row of the pointer variable "inputMatrix".
		*currentRow = *currentRow + 1;
    }
    else {
        for (i = l; i <= r; i++) {
            // swap data
            dataHolder = inputMatrix[l];
            inputMatrix[l] = inputMatrix[i];
            inputMatrix[i] = dataHolder;
            
            // Apply recursive permutation
            getPermutations(l+1, r, m, currentRow, inputMatrix);
            
            // make a backtrack swap
            dataHolder = inputMatrix[l];
            inputMatrix[l] = inputMatrix[i];
            inputMatrix[i] = dataHolder;
        }
    }
}


/**
* The "getSimpleLinearRegression()" function is used to apply the
* machine learning algorithm called simple linear regression. Within
* this process, the best fitting equation with the form of "y_hat =
* b_0 + b_1*x" will be identified with respect to the sampled data
* given through the argument pointer variables "X" and "Y". As a
* result, the identified coefficient values will be stored in the
* argument pointer variable "b".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INNITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m=1" 'DOUBLE' MEMORY
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
* LAST UPDATE: N/A
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
*					 AND INNITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m=1" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE IS ALLOCATED AND INNITIALIZED
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
*						 TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 13, 2021
* LAST UPDATE: N/A
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
}











/*

NOTE: The algorithm section that applied the matrix inverse using the Gauss-Jordan method was inspired in the following source:
"CodeSansar. Matrix Inverse Using Gauss Jordan Method C Program. November 16, 2021 (Recovery date), de CodeSansar Sitio web: https://bit.ly/3CowwSy".

NOTE: "b" will temporarly be have to be allocated with "m+1" columns and an equivalent of the factorial of "(m+1)" for its rows.
	  In addition, "b" will store the coefficients for all the permutations idetified, were each of its coefficiets per permutation
	  will be stored in a different row inside "b".

*/






/**
* The "getSimpleLinearRegression()" function is used to apply the
* machine learning algorithm called simple linear regression. Within
* this process, the best fitting equation with the form of "y_hat =
* b_0 + b_1*x" will be identified with respect to the sampled data
* given through the argument pointer variables "X" and "Y". As a
* result, the identified coefficient values will be stored in the
* argument pointer variable "b".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INNITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m=1" 'DOUBLE' MEMORY
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
* CREATION DATE: NOVEMBER XX, 2021
* LAST UPDATE: N/A
*/
void getMultipleLinearRegression(double *X_tilde, double *Y, int n, int m, int p, char isVariableOptimizer, double *b) {
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
	
	// Obtain the number of possible permutations (which will be the factorial of "(m+1)").
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	int factorialValue = mPlusOne;
	for (int i=1; i<mPlusOne; i++) {
		factorialValue = factorialValue*i;
	}
	// Innitialize the data of the first row of the pointer variable that will store all the column indexes permutations that the matrix "X_tilde" can have.
	int *columnPermutations = (int *) malloc(factorialValue*mPlusOne*sizeof(int));
	for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) { // The initialized value for the first row of the matrix "columnPermutations" will represent the original column index arrangement.
		columnPermutations[currentColumn] = currentColumn;
	}
	// Get all possible index permutations with respect to the data stored in the first row of "columnPermutations".
	int current_Permutation = 0;
	getPermutations(0, m, mPlusOne, &current_Permutation, columnPermutations);
	
	// Apply the desired machine learning algorithm over each permutation identified.
	int currentRowTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentPermutationTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	double *new_X_tilde = (double *) malloc(n*mPlusOne*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the data from which the desired machine learning method will be calcualted.
	double *TransposeOf_new_X_tilde = (double *) malloc(mPlusOne*n*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	double *matMul1 = (double *) malloc(mPlusOne*mPlusOne*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between "new_X_tilde" and its transpose.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentColumnTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	double ratioModifier; // This variable is used to store the ratio modifier for the current row whose values will be updated due to the inverse matrix method.
	int currentColumnTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn2; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	double *matMul2 = (double *) malloc(mPlusOne*n*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the result of making a matrix multiplication between the resulting inverse matrix of this process and the transpose of the matrix "new_X_tilde".
	for (int currentPermutation=0; currentPermutation<factorialValue; currentPermutation++) {
		// Fill "new_X_tilde" with the data of X_tilde but arranged with the current permutation indicated in "columnPermutations". In addition, we obtain the transpose of "new_X_tilde".
		currentColumn2 = 0; // We reset the counters used in the following for-loop.
		for (int currentRow=0; currentRow<n; currentRow++) {
			currentRow2 = 0; // We reset the counters used in the following for-loop.
			currentRowTimesMplusOne = currentRow*mPlusOne;
			currentPermutationTimesMplusOne = currentPermutation*mPlusOne;
			for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
				currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
		        new_X_tilde[currentRowAndColumn] = X_tilde[columnPermutations[currentColumn + currentPermutationTimesMplusOne] + currentRowTimesMplusOne];
		        TransposeOf_new_X_tilde[currentColumn2 + currentRow2*n] = new_X_tilde[currentRowAndColumn]; 
				currentRow2++;
		    }
		    currentColumn2++;
		}
		
		// In order to start obtaining the coefficients, we multiply the matrix "TransposeOf_new_X_tilde" with the matrix "new_X_tilde" and store the result in the matrix "matMul1".
		for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
			currentRowTimesMplusOne = currentRow*mPlusOne;
			currentRowTimesN = currentRow*n;
			for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
				currentColumnTimesN = currentColumn*n;
				currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
				matMul1[currentRowAndColumn] = 0; // reset the value that will be used for the current matrix multiplication.
				for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
					// Here we want to multiply "TransposeOf_new_X_tilde" with the matrix "new_X_tilde", but we will use "TransposeOf_new_X_tilde" for such multiplication since they contain the same data, for performance purposes.
					matMul1[currentRowAndColumn] = matMul1[currentRowAndColumn] + TransposeOf_new_X_tilde[currentMultipliedElements + currentRowTimesN] * TransposeOf_new_X_tilde[currentMultipliedElements + currentColumnTimesN];
				}
		    }
		}
		
		// In order to continue obtaining the coefficients, we innitialize the data of a unitary matrix with the same dimensions of the matrix "matMul1".
		// NOTE: Because getting the data of the transpose of "new_X_tilde"
		//		 directly from that same variable ("new_X_tilde"), will
		//		 increase performance, we will store the matrix inverse of
		//		 "matMul1" in the variable "TransposeOf_new_X_tilde", in
		//		 order to maximize computational resources and further
		//		 increase performance by not having to allocate more memory
		//		 in the computer system.
		for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // We set all values to zero.
			for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
				TransposeOf_new_X_tilde[currentColumn + currentRow*mPlusOne] = 0;
		    }
		}
		for (int currentUnitaryValue=0; currentUnitaryValue<mPlusOne; currentUnitaryValue++) { // We set the corresponding 1's values to make the corresponding unitary matrix.
			TransposeOf_new_X_tilde[currentUnitaryValue + currentUnitaryValue*mPlusOne] = 1;
		}
		
		// In order to continue obtaining the coefficients, we calculate the matrix inverse of "matMul1" with the Gauss-Jordan approach.
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
						TransposeOf_new_X_tilde[currentRowAndColumn] = TransposeOf_new_X_tilde[currentRowAndColumn] - ratioModifier * TransposeOf_new_X_tilde[currentModifiedElements + currentColumnTimesMplusOne];
					}
				}
			}
	    }
		for (int currentRow=0; currentRow<mPlusOne; currentRow++) { // We apply the last step of the approach used in order to obtain the diagonal of 1's in the principal matrix.
			currentRowTimesMplusOne = currentRow*mPlusOne;
			currentRowAndColumn2 = currentRow + currentRowTimesMplusOne;
			for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
				currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
				TransposeOf_new_X_tilde[currentRowAndColumn] = TransposeOf_new_X_tilde[currentRowAndColumn] / matMul1[currentRowAndColumn2];
			}
	    }
	    
		// In order to continue obtaining the coefficients, we multiply the inverse matrix that was obtained by the transpose of the matrix "new_X_tilde".
		// NOTE: Remember that we will get the data of the transpose of
		//		 "new_X_tilde" directly from that same variable
		//		 ("new_X_tilde") due to performance reasons and; that the
		//		 inverse matrix that was obtained is stored in
		//		 "TransposeOf_new_X_tilde".
		for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
			currentRowTimesN = currentRow*n;
			currentRowTimesMplusOne = currentRow*mPlusOne;
			for (int currentColumn=0; currentColumn<n; currentColumn++) {
				currentRowAndColumn = currentColumn + currentRowTimesN;
				currentColumnTimesMplusOne = currentColumn*mPlusOne;
				matMul2[currentRowAndColumn] = 0; // reset the value that will be used for the current matrix multiplication.
				for (int currentMultipliedElements=0; currentMultipliedElements<mPlusOne; currentMultipliedElements++) {
					matMul2[currentRowAndColumn] = matMul2[currentRowAndColumn] + TransposeOf_new_X_tilde[currentMultipliedElements + currentRowTimesMplusOne] * new_X_tilde[currentMultipliedElements + currentColumnTimesMplusOne];
				}
		    }
		}
		
		// In order to conclude obtaining the coefficients ("b"), we multiply the previously resulting matrix ("matMul2") by the output matrix "Y".
		for (int currentColumn=0; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentPermutation*mPlusOne;
			currentRowTimesN = currentColumn*n;
			for (int currentMultipliedElements=0; currentMultipliedElements<n; currentMultipliedElements++) {
				b[currentRowAndColumn] = b[currentRowAndColumn] + matMul2[currentMultipliedElements + currentRowTimesN] * Y[currentMultipliedElements];
			}
		}
	}
	
	// Free the Heap memory used for the locally allocated variables since they will no longer be used.
	free(columnPermutations);
	free(new_X_tilde);
	free(TransposeOf_new_X_tilde);
	free(matMul1);
	free(matMul2);
}

