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
#include "CenyMLstatistics.h"
// IMPORTANT NOTE: This library uses the math.h library and therefore,
//				   remember to use the "-lm" flag when compiling it.


/**
* The "getMean()" function is used to calculate the mean of all the
* columns of the input matrix and its result is stored as a vector
* in its pointer argument variable "mean".
* 
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   input matrix, from which the mean
*							   will be calculated. THIS VARIABLE
*							   SHOULD BE ALLOCATED AND INNITIALIZED
*							   BEFORE CALLING THIS FUNCTION WITH
*							   A SIZE OF "n" TIMES "m" 'DOUBLE'
*							   MEMORY SPACES.
*
* @param int n - This argument will represent the total number
*				 of samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number
*				 of features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param double *mean - This argument will contain the pointer to
*						a memory allocated variable in which we
*						will store the mean for each column
*						contained in the argument variable
*					    "inputMatrix". IT IS INDISPENSABLE THAT THIS
*						VARIABLE IS ALLOCATED BEFORE CALLING THIS
*						FUNCTION AND INNITIALIZED WITH 0s AND A
*						VARIABLE SIZE OF "m" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "mean".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: SEPTEMBER 23, 2021
* LAST UPDATE: NOVEMBER 09, 2021
*/
void getMean(double *inputMatrix, int n, int m, double *mean) {
	// We obtain the mean for each of the columns of the input matrix.
    int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesM = currentRow*m;
    	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
    		mean[currentColumn] += inputMatrix[currentColumn + currentRowTimesM];
		}
	}
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		mean[currentColumn] = mean[currentColumn]/n;
	}
}


/**
* The "getSort()" function is used to apply a sort method in
* ascending order to all the rows contained in each of the
* columns of the input argument matrix whose pointer is
* "inputMatrix".
* 
* @param char desiredSortMethod[] - This argument will contain the
*							   		string or array of characters
*									that will specify the sort
*									method requested by the
*									implementer. Its possible values
*									are the following:
*									1) "quicksort" = applies the
*													 quicksort method.
*
* @param int n - This argument will represent the total number
*				 of samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number
*				 of features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated input
*							   matrix, from which a certain sort
*							   method will be applied. THIS VARIABLE
*							   SHOULD BE ALLOCATED AND INNITIALIZED
*							   BEFORE CALLING THIS FUNCTION WITH
*							   A SIZE OF "n" TIMES "m" 'DOUBLE'
*							   MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: OCTOBER 07, 2021
* LAST UPDATE: NOVEMBER 09, 2021
*/
//TODO: Test the performance of the "quick sort" method when argumenting the entire matrix in "applyQuickSort()" instead of several vectors, to compare it with the current method.
void getSort(char desiredSortMethod[], int n, int m, double *inputMatrix) {
	// If the implementer requested the "quick sort" method, then apply it through the following code.
	if (strcmp(desiredSortMethod, "quicksort") == 0) {
		// We allocate the needed memory for the vector containing the rows of a certain column from the "inputMatrix".
		double *rowsOfCurrentColumn = (double *) malloc(n*sizeof(double));
		
		// We apply the recursive function that will implement the "quick sort" method but for each column seperately.
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			// We pass the values of the current column to be evaluated to the pointer variable "rowsOfCurrentColumn".
			for (int currentRow = 0; currentRow < n; currentRow++) {
				rowsOfCurrentColumn[currentRow] = inputMatrix[currentColumn + currentRow*m];
			}
			
			// We start the "quick sort" function for the current column of data.
			applyQuickSort(0, n-1, rowsOfCurrentColumn);
			
			// We store the sorted column values back into the corresponding memory location of the pointer "inputMatrix".
			for (int currentRow = 0; currentRow < n; currentRow++) {
				inputMatrix[currentColumn + currentRow*m] = rowsOfCurrentColumn[currentRow];
			}
		}
		
		// Free the Heap memory used for the currently allocated variables.
		free(rowsOfCurrentColumn);
	}
}
/**
* The "applyQuickSort()" function is inspired in the code made and
* explained by "latincoder" in the Youtube video "Algoritmo Quicksort
* en lenguaje C -Tutorial Implementación recursiva"
* (URL = https://www.youtube.com/watch?v=rADlgxPQa_w). Nonetheless,
* as coded in this file, this function is used to apply the "quick
* sort" method to sort the values in ascending order of the vector
* whose pointer is the argument variable "inputVector".
* 
* @param int minIndexLimit - This argument will represent the minimum
*							 index value from which the left elements
*							 will start being addressed for the
*							 current "quick sort" requested.
*
* @param int maxIndexLimit - This argument will represent the maximum
*							 index value from which the right elements
*							 will start being addressed for the
*							 current "quick sort" requested.
*
* @param double *inputVector - This argument will contain the
*							   pointer to a memory allocated input
*							   vector, from which the "quick sort"
*							   method will be applied. THIS VARIABLE
*							   SHOULD BE ALLOCATED AND INNITIALIZED
*							   BEFORE CALLING THIS FUNCTION WITH
*							   A SIZE OF "n" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputVector".
* NOTE: In order for the implementer to understand this function, it
*		is expected that he/she already knows/masters the theory of
*		the "quick sort" method.
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: OCTOBER 07, 2021
* LAST UPDATE: NOVEMBER 08, 2021
*/
static void applyQuickSort(int minIndexLimit, int maxIndexLimit, double *inputVector) {
	// We will define the local variables to be used in this function.
	int indexOfLeftElement = minIndexLimit;
	int indexOfRightElement = maxIndexLimit;
	double pivotValue = inputVector[(indexOfLeftElement + indexOfRightElement)/2];
	double exchangeValue;
	
	// We apply once the "quick sort" method through this while loop.
	while (indexOfLeftElement <= indexOfRightElement) {
		// We identify the index of the left elements where its value is greater than the pivot value if available.
		while ((inputVector[indexOfLeftElement]<pivotValue) && (indexOfLeftElement<maxIndexLimit)) {
			indexOfLeftElement++;
		}
		
		// We identify the index of the right elements where its value is smaller than the pivot value if available.
		while ((pivotValue<inputVector[indexOfRightElement]) && (indexOfRightElement>minIndexLimit)) {
			indexOfRightElement--;
		}
		
		// If the current 'left index' is smaller or equatl to the 'right index', then exchange their values.
		if (indexOfLeftElement <= indexOfRightElement) {
			exchangeValue = inputVector[indexOfLeftElement];
			inputVector[indexOfLeftElement] = inputVector[indexOfRightElement];
			inputVector[indexOfRightElement] = exchangeValue;
			indexOfLeftElement++;
			indexOfRightElement--;
		}
	}
	
	// If there are still more solutions to be made, then apply this function recursively again.
	if (minIndexLimit < indexOfRightElement) {
		applyQuickSort(minIndexLimit, indexOfRightElement, inputVector);
	}
	if (maxIndexLimit > indexOfLeftElement) {
		applyQuickSort(indexOfLeftElement, maxIndexLimit,inputVector);
	}
}


/**
* The "getMedian()" function is used to calculate the median of all
* the columns of the input matrix and its result is stored as a vector
* in its pointer argument variable "Q2".
* 
* @param char desiredSortMethod[] - This argument will contain the
*							   		string or array of characters
*									that will specify the sort
*									method requested by the
*									implementer. Its possible values
*									are the following:
*									1) "quicksort" = applies the
*													 quicksort method.
*
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   input matrix, from which the
*							   median will be calculated. THIS
*							   VARIABLE SHOULD BE ALLOCATED AND
*							   INNITIALIZED BEFORE CALLING THIS
*							   FUNCTION WITH A SIZE OF "n" TIMES "m"
*							   'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number
*				 of samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number
*				 of features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param double *Q2 - This argument will contain the pointer to
*					  a memory allocated variable in which we
*					  will store the median for each column
*					  contained in the argument variable
*					  "inputMatrix". IT IS INDISPENSABLE THAT THIS
* 					  VARIABLE IS ALLOCATED BEFORE CALLING THIS
*					  FUNCTION WITH A SIZE OF "m" 'DOUBLE' MEMORY
*					  SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "Q2".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: OCTOBER 11, 2021
* LAST UPDATE: NOVEMBER 09, 2021
*/
void getMedian(char desiredSortMethod[], double *inputMatrix, int n, int m, double *Q2) {
	// If the implementer requested the "quick sort" method for the sorting process, then apply it through the following code.
	if (strcmp(desiredSortMethod, "quicksort") == 0) {
		// We allocate the needed memory for the vector containing the rows of a certain column from the "inputMatrix".
		double *rowsOfCurrentColumn = (double *) malloc(n*sizeof(double));
		
		// Sort the column values of the input matrix and then, find/identify the median of each column depending on whether n is even or odd.
		if ((n) == ((n/2)*2)) { // Because n is even, apply the following solution:
			// We declared local variables to be required.
			int nMinusOneDividedByTwo = (n-1)/2; // This variable is used to store a repetitive mathematical operation, for performance purposes.
			
			// We apply the recursive function that will implement the "quick sort" method but for each column seperately.
			for (int currentColumn = 0; currentColumn < m; currentColumn++) {
				// We pass the values of the current column to be evaluated to the pointer variable "rowsOfCurrentColumn".
				for (int currentRow = 0; currentRow < n; currentRow++) {
					rowsOfCurrentColumn[currentRow] = inputMatrix[currentColumn + currentRow*m];
				}
				
				// We start the "quick sort" function for the current column of data.
				applyQuickSort(0, n-1, rowsOfCurrentColumn);
				
				// We find/identify the median or Q2 for the sorted current column of the input matrix and store its value.
				Q2[currentColumn] = (rowsOfCurrentColumn[nMinusOneDividedByTwo] + rowsOfCurrentColumn[nMinusOneDividedByTwo + 1])/2;
			}
		} else { // Because n is odd, apply the following solution:
			// We apply the recursive function that will implement the "quick sort" method but for each column seperately.
			for (int currentColumn = 0; currentColumn < m; currentColumn++) {
				// We pass the values of the current column to be evaluated to the pointer variable "rowsOfCurrentColumn".
				for (int currentRow = 0; currentRow < n; currentRow++) {
					rowsOfCurrentColumn[currentRow] = inputMatrix[currentColumn + currentRow*m];
				}
				
				// We start the "quick sort" function for the current column of data.
				applyQuickSort(0, n-1, rowsOfCurrentColumn);
				
				// We find/identify the median or Q2 for the sorted current column of the input matrix and store its value.
				Q2[currentColumn] = rowsOfCurrentColumn[n/2];
			}
		}
		
		// Free the Heap memory used for the currently allocated variables.
		free(rowsOfCurrentColumn);
	}
}


/**
* The "getVariance()" function is used to calculate the variance of all the
* columns of the input matrix and its result is stored as a vector
* in its pointer argument variable "variance".
* 
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   input matrix, from which the variance
*							   will be calculated. THIS VARIABLE
*							   SHOULD BE ALLOCATED AND INNITIALIZED
*							   BEFORE CALLING THIS FUNCTION WITH
*							   A SIZE OF "n" TIMES "m" 'DOUBLE'
*							   MEMORY SPACES.
*
* @param int n - This argument will represent the total number
*				 of samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number
*				 of features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int degreesOfFreedom - This argument will represent the desired
*								value for the degrees of freedom to be
*								applied when calculating the variance.
*
* @param double *variance - This argument will contain the pointer to
*							a memory allocated variable in which we
*							will store the variance for each column
*							contained in the argument variable
*					    	"inputMatrix". IT IS INDISPENSABLE THAT
*							THIS VARIABLE IS ALLOCATED BEFORE CALLING
*							THIS FUNCTION AND INNITIALIZED WITH 0 AND A
*							SIZE OF "m" 'DOUBLE MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "variance".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: OCTOBER 17, 2021
* LAST UPDATE: NOVEMBER 09, 2021
*/
void getVariance(double *inputMatrix, int n, int m, int degreesOfFreedom, double *variance) {
	// We declare and innitialize the variable "mean" with "0"s. The mean for each column will be stored in this variable.
	double mean[m]; // The mean of each column will be stored in this variable.
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		mean[currentColumn] = 0;
	}
	
	// We obtain the mean for each of the columns of the input matrix.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
    	currentRowTimesM = currentRow*m;
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
    		mean[currentColumn] += inputMatrix[(currentColumn + currentRowTimesM)];
		}
	}
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		mean[currentColumn] = mean[currentColumn]/n;
	}
	
	// We obtain the variance for each of the columns of the input matrix.
	double squareThisValue; // Variable used to store the value that wants to be squared.
    for (int currentRow = 0; currentRow < n; currentRow++) {
    	currentRowTimesM = currentRow*m;
    	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
    		squareThisValue = inputMatrix[(currentColumn + currentRowTimesM)] - mean[currentColumn];
    		variance[currentColumn] += squareThisValue*squareThisValue;
		}
	}
	degreesOfFreedom = n - degreesOfFreedom; // We apply the requested degrees of freedom.
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		variance[currentColumn] = variance[currentColumn]/degreesOfFreedom;
	}
}


/**
* The "getStandardDeviation()" function is used to calculate the
* standard deviation of all the columns of the input matrix and its
* result is stored as a vector in its pointer argument variable
* "standardDeviation".
* 
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   input matrix, from which the standard
*							   deviation will be calculated. THIS
*							   VARIABLE SHOULD BE ALLOCATED AND
*							   INNITIALIZED BEFORE CALLING THIS
*							   FUNCTION WITH A SIZE OF "n" TIMES "m"
*							   'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number
*				 of samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number
*				 of features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int degreesOfFreedom - This argument will represent the desired
*								value for the degrees of freedom to be
*								applied when calculating the standard
*								deviation.
*
* @param double *standardDeviation - This argument will contain the
*									 pointer to a memory allocated
*									 variable in which we will store
*									 the standard deviation for each
*									 column contained in the argument
*									 variable "inputMatrix". IT IS
*									 INDISPENSABLE THAT THIS VARIABLE
*									 IS ALLOCATED BEFORE CALLING THIS
*									 FUNCTION AND INNITIALIZED WITH 0
*									 AND A SIZE OF "m" 'DOUBLE' MEMORY
*									 SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "standardDeviation".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: OCTOBER 18, 2021
* LAST UPDATE: NOVEMBER 09, 2021
*/
void getStandardDeviation(double *inputMatrix, int n, int m, int degreesOfFreedom, double *standardDeviation) {
	// We declare and innitialize the variable "mean" with "0"s. The mean for each column will be stored in this variable.
	double mean[m]; // The mean of each column will be stored in this variable.
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		mean[currentColumn] = 0;
	}
	
	// We obtain the mean for each of the columns of the input matrix.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
    	currentRowTimesM = currentRow*m;
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
    		mean[currentColumn] += inputMatrix[(currentColumn + currentRowTimesM)];
		}
	}
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		mean[currentColumn] = mean[currentColumn]/n;
	}
	
	// We calculate the variance and then obtain its square root to get the standard deviation for each of the columns of the input matrix.
    double squareThisValue; // Variable used to store the value that wants to be squared.
    for (int currentRow = 0; currentRow < n; currentRow++) {
    	currentRowTimesM = currentRow*m;
    	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
    		squareThisValue = inputMatrix[(currentColumn + currentRowTimesM)] - mean[currentColumn];
    		standardDeviation[currentColumn] += squareThisValue*squareThisValue;
		}
	}
	degreesOfFreedom = n-degreesOfFreedom; // We apply the requested degrees of freedom.
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		standardDeviation[currentColumn] = sqrt(standardDeviation[currentColumn]/degreesOfFreedom);
	}
}


/**
* The "getQuickMode()" function is used to apply a mode method
* to each column of the input argument matrix whose pointer is
* "inputMatrix".
* 
* @param char desiredSortMethod[] - This argument will contain the
*							   		string or array of characters
*									that will specify the sort
*									method requested by the
*									implementer. Its possible values
*									are the following:
*									1) "quicksort" = applies the
*													 quicksort method.
*
* @param int n - This argument will represent the total number
*				 of samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number
*				 of features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated input
*							   matrix that will be copied into a
*							   local variable to then process it
*							   and obtain its mode. THIS VARIABLE
*							   SHOULD BE ALLOCATED AND INNITIALIZED
*							   BEFORE CALLING THIS FUNCTION WITH
*							   A SIZE OF "n" TIMES "m" 'DOUBLE'
*							   MEMORY SPACES.
*
* @param int *Mo_n - This argument will contain the pointer to a
*					 memory allocated variable in which each "k-th"
*					 column will store the rows length value of the
*					 "k-th" column of the argument variable "Mo".
*					 THIS VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
*                    WITH ZEROS BEFORE CALLING THIS FUNCTION WITH A
*                    SIZE OF "n" 'DOUBLE' MEMORY SPACES.
*
* @param double *Mo - This argument will contain the pointer to a
*					  memory allocated variable in which the sort
*					  result obtained will be stored. The mode
*					  obtained will be with respect to each column,
*					  where either the single or multiple mode values
*					  identified for a certain column of the argument
*					  variable "inputMatrix" will be all stored in the
*					  same index column in "Mo". THIS VARIABLE SHOULD
*					  BE ALLOCATED BEFORE CALLING THIS FUNCTION WITH
*					  A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* NOTE: THE MODE RESULT IS STORED IN THE MEMORY ALLOCATED POINTER
*		VARIABLE "Mo" AND THE LENGTH OF THE ROWS FOR EACH OF ITS
*		COLUMNS ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Mo_n".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: OCTOBER 19, 2021
* LAST UPDATE: NOVEMBER 18, 2021
*/
//TODO: Test the performance of the "quick sort" method when argumenting the entire matrix in "applyQuickSort()" instead of several vectors, to compare it with the current method.
void getQuickMode(char desiredSortMethod[], int n, int m, double *inputMatrix, int *Mo_n, double *Mo) {
	// If the implementer requested the "quick sort" method, then apply it through the following code.
	if (strcmp(desiredSortMethod, "quicksort") == 0) {
		// We declared and allocate the required memory.
		int indexBeingCounted; // This variable is used to determine the index of the input sorted vector that is currently being counter in the mode process.
		int maxRepeatitions; // This variable is used to store the maximum number of times that a data in the input sorted vector was repeated.
		int currentMo_n; // This variable is used to determine the current row of "Mo".
		double *rowsOfCurrentColumn = (double *) malloc(n*sizeof(double)); // This pointer variable will be used as the vector containing the rows of a certain column from the "inputMatrix".
		double *counterOfInputVector = (double *) calloc(n, sizeof(double)); // This pointer variable will be used as a repetition counter for the mode to be calculated.
		
		// We apply the recursive function that will implement the "quick sort" method but for each column seperately.
		for (int currentColumn=0; currentColumn < m; currentColumn++) {
			// We pass the values of the current column to be evaluated to the pointer variable "rowsOfCurrentColumn".
			for (int currentRow = 0; currentRow < n; currentRow++) {
				rowsOfCurrentColumn[currentRow] = inputMatrix[currentColumn + currentRow*m];
			}
			
			// We start the "quick sort" function for the current column of data.
			applyQuickSort(0, n-1, rowsOfCurrentColumn);
			
			// Reset these variables for the next vector from which the mode will be obtained.
			indexBeingCounted = 0;
			maxRepeatitions = 0;
			currentMo_n = 0;
			for (int currentRow=0; currentRow < n; currentRow++) {
				counterOfInputVector[currentRow] = 0;
			}
			
			// We count the number of times each data of the input vector gets repeated.
			for (int currentRow=1; currentRow < n; currentRow++) {
				if (rowsOfCurrentColumn[indexBeingCounted] == rowsOfCurrentColumn[currentRow]) { // if the current index being counted matches the current row being inspected, then increase the counter of such index by 1.
					counterOfInputVector[indexBeingCounted] = counterOfInputVector[indexBeingCounted] + 1;
					
					// We obtain the maximum number of times that any of the data of the input vector gets repeated.
					if (counterOfInputVector[indexBeingCounted] > maxRepeatitions) {
						maxRepeatitions = counterOfInputVector[indexBeingCounted];
					}
				} else { // If the current index being counted did not matched the current row being inspected, then change the current index being counted to the one being indicated in the current row.
					indexBeingCounted = currentRow;
				}
			}
			
			// We obtain the mode.
			for (int currentRow=0; currentRow < n; currentRow++) {
				if (counterOfInputVector[currentRow] == maxRepeatitions) { // If current value of the input vector has the maximum number of repetitions, then add it into the mode result.				
					Mo[currentColumn + currentMo_n*m] = rowsOfCurrentColumn[currentRow]; // We store the mode in the pointer variable "Mo"
					currentMo_n++; // We increase this index to save a new value detected to be mode if available.
					Mo_n[currentColumn] = Mo_n[currentColumn] + 1; // We also store the number of rows that the current column will have in the pointer variable "Mo_n".
				}
			}
		}
		
		// Free the Heap memory used for the currently allocated variables.
		free(rowsOfCurrentColumn);
		free(counterOfInputVector);
	}
}

