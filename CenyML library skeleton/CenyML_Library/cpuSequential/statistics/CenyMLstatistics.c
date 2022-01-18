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
* LAST UPDATE: DECEMBER 04, 2021
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
	
	return;
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
* LAST UPDATE: DECEMBER 04, 2021
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
	
	return;
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
* LAST UPDATE: DECEMBER 04, 2021
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
	
	return;
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
* LAST UPDATE: DECEMBER 04, 2021
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
	
	return;
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
* LAST UPDATE: DECEMBER 04, 2021
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
	
	return;
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
* LAST UPDATE: DECEMBER 04, 2021
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
	
	return;
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
* LAST UPDATE: DECEMBER 04, 2021
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
	
	return;
}


/**
* The "getMeanIntervals()" function is used to calculate the mean
* intervals with respect to the given mean(s); standard deviation(s)
* and; the desired trust interval and, its result is stored in its
* pointer argument variable "meanIntervals". In addition, the
* implementer can choose whether to use the standard normal
* distribution (z distribution) or the t distribution in order to
* make this function calculate the area under the normal curve
* with the most convenient option.

* IMPORTANT NOTE: The way the mean intervals are calculated by this
* function is just as in the book "Probabilidad y estadistica para
* ingenieria y ciencia 8va edicion (2007)"  of Walpole Myers Myers
* Ye, as explained in the pages 275, 277 and 279., which is made by
* using the "A.3" and "A.4" appendix tables of that book.
*
* 
* @param double *mean - This argument will contain the pointer to
*						a memory allocated variable in which the
*						mean(s) is(are) expected to be already
*						stored. In addition, each column of this
*						variable will be considered as a different
*						mean. IT IS INDISPENSABLE THAT THIS VARIABLE
*						IS ALLOCATED BEFORE CALLING THIS FUNCTION AND
*						INNITIALIZED WITH ITS CORRESPONDING MEAN
*						VALUES AND A VARIABLE SIZE OF "m" 'DOUBLE'
*						MEMORY SPACES.
*
* @param double *standardDeviation - This argument will contain the
*						pointer to a memory allocated variable in
*						which the standard deviation(s) is(are)
*						expected to be already stored. In addition,
*						each column of this variable will be
*						considered as a different standard deviation.
*						IT IS INDISPENSABLE THAT THIS VARIABLE IS
*						ALLOCATED BEFORE CALLING THIS FUNCTION AND
*						INNITIALIZED WITH ITS CORRESPONDING STANDARD
*						DEVIATION VALUES AND A VARIABLE SIZE OF "m"
*						'DOUBLE' MEMORY SPACES.
*
* @param float desiredTrustInterval - This argument will represent the
*									  desired trust interval to be
*									  calculated and used for the
*									  calculation of the corresponding
*									  mean intervals.
*
* @param char isTdistribution = This argument variable will work as a
*						  		flag to indicate whether it is desired
*								to calculate the area under the normal
*								curve by using the standard normal
*								distribution (z distribution) or the t
*								distribution. The following will list
*								the possible valid outcomes for this
*								variable:
*					  		    1) "isTdistribution" = (int) 0:
*									This function will calculate the
*									area under a normal curve by using
*									the standard normal distribution
*									(z distribution) method. Remember,
*									that this method is used when the
*									true standard deviation is known.
*									However, it is often used when the
*									true standard deviation is not
*									known but where the samples used
*									for its calculation were equal or
*									greater than 30 (because beyond
*									that number of samples, the
*									difference is very little between
*									the z distribution and the t
*									distribution).
*					  		    2) "isTdistribution" = (int) 1:
*									This function will calculate the
*									area under a normal curve by using
*									the t distribution method. Remember,
*									that this method is used when the
*									true standard deviation is not known.
*									IMPORTANT NOTE: FOR NOW, THIS
*									FUNCTION WILL ONLY BE ABLE TO
*									CALCULATE UP TO "n=<31" because other
*									values have yet to be programmed.
*
* @param int n - This argument will represent the total number
*				 of samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number
*				 of features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param double *meanIntervals - This argument will contain the pointer
*					to a memory allocated variable in which we will
*					store the mean intervals with respect to the values
*					of each independent variable (column) contained in
*					the argument variables "mean" and
*					"standardDeviation". In regards to this, both the
*					lower and upper mean intervals, corresponding to
*					the first independent variable from the argument
*					variables "mean" and "standardDeviation", will be
*					stored in the row index 0 of the argument variable
*					"meanIntervals". The lower and upper mean intervals
*					of the second independent variable of "mean" and
*					"standardDeviation" will be stored in the row index
*					1 of "meanIntervals" and so on up until the lower
*					and upper mean intervals of the last independent
*					variable of "mean" and "standardDeviation" will be
*					stored in the row index "m-1" of "meanIntervals".
*					In addition, for all the rows in "meanIntervals",
*					the column index 0 will contain the lower mean
*					interval and its column index 1 will contain the
*					upper mean interval. FINALLY, IT IS INDISPENSABLE
*					THAT THIS VARIABLE IS ALLOCATED BEFORE CALLING
*					THIS FUNCTION WITH A VARIABLE SIZE OF "m" TIMES
*					"2" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "meanIntervals".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 11, 2021
* LAST UPDATE: N/A
*/
void getMeanIntervals(double *mean, double *standardDeviation, float desiredTrustInterval, char isTdistribution, int n, int m, double *meanIntervals) {
	// If the requested desired trust interval is not within the options that the CenyML library can calculate, then emit an
	// error message and terminate the program. Otherwise, continue with the program.
	if ((desiredTrustInterval!=95) && (desiredTrustInterval!=99) && (desiredTrustInterval!=(float)(99.9))) {
		printf("\nERROR: The requested trust interval has not yet been programmed in this function. Please select one of the available ones: 95, 99 or 99.9.\n");
		exit(1);
	}
	// If the samples are less than value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (n < 2) {
		printf("\nERROR: The defined samples must be equal or greater than 2 for this particular algorithm.\n");
		exit(1);
	}
	// If the independent variables are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m < 1) {
		printf("\nERROR: The independent variables must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the "t distribution" method was chosen and "n" is greater than 31, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((isTdistribution==1) && (n>31)) {
		printf("\nERROR: When the t distribution is chosen for this algorithm, then the defined samples must be less than 32. This is because t values with samples equal or greater than 32 have yet to be added into this library.\n");
		exit(1);
	}
	// If "t distribution" is different from "1" or "0", then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((isTdistribution!=1) && (isTdistribution!=0)) {
		printf("\nERROR: An invalid value for the argument variable \"isTdistribution\" has been defined. Please store a value of either (int) 1 or (int) 0.\n");
		exit(1);
	}
	
	// Calculate the area under the normal curve with the requested method.
	double areaUnderNormalCurve; // This variable will store the area under the normal curve (which will be either with the "z" or the "t" distribution).
	if (isTdistribution == 1) {
		// We calculate the t distribution (t_{alpha/2}).
		if (desiredTrustInterval == 95) { // If the desired trust interval is 95%, then get the following t distribution (t_{alpha/2}).
			double tValue[] = {12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.160,2.145,2.131,2.120,2.110,2.101,2.093,2.086,2.080,2.074,2.069,2.064,2.060,2.056,2.052,2.048,2.045,2.042};
			areaUnderNormalCurve = tValue[n-2]; // We retrieve and store the corresponding "t" value.
		} else if (desiredTrustInterval == 99) { // If the desired trust interval is 99%, then get the following t distribution (t_{alpha/2}).
			double tValue[] = {63.656,9.925,5.841,4.604,4.032,3.707,3.499,3.355,3.250,3.196,3.106,3.055,3.012,2.977,2.947,2.921,2.898,2.878,2.861,2.845,2.831,2.819,2.807,2.797,2.787,2.779,2.771,2.763,2.756,2.750};
			areaUnderNormalCurve = tValue[n-2]; // We retrieve and store the corresponding "t" value.
		} else { // If the desired trust interval is 99.9%, then get the following t distribution (t_{alpha/2}).
			double tValue[] = {636.578,31.600,12.924,8.610,6.869,5.959,5.408,5.041,4.781,4.587,4.437,4.318,4.221,4.140,4.073,4.015,3.965,3.922,3.883,3.850,3.819,3.792,3.768,3.745,3.725,3.707,3.689,3.674,3.660,3.646};
			areaUnderNormalCurve = tValue[n-2]; // We retrieve and store the corresponding "t" value.
		}
	} else {
		// We calculate the standard normal distribution (z_{alpha/2}}.
		if (desiredTrustInterval == 95) { // If the desired trust interval is 95%, then get the following z distribution (z_{alpha/2}).
			areaUnderNormalCurve = 1.96;
		} else if (desiredTrustInterval == 99) { // If the desired trust interval is 99%, then get the following z distribution (z_{alpha/2}).
			areaUnderNormalCurve = 2.576;
		} else { // If the desired trust interval is 99.9%, then get the following z distribution (z_{alpha/2}).
			areaUnderNormalCurve = 3.29;
		}
	}
	
	// Calculate the mean intervals.
	int currentRowTimesTwo; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	double squareRootOfN = sqrt(n); // This variable is used to store a repetitively used value, for performance purposes.
	for (int currentRow=0; currentRow<m; currentRow++) {
    	currentRowTimesTwo = currentRow * 2;
		meanIntervals[currentRowTimesTwo] = mean[currentRow] - areaUnderNormalCurve * standardDeviation[currentRow] / squareRootOfN; // We calculate and store the lower mean intervals that were requested.
    	meanIntervals[1 + currentRowTimesTwo] = mean[currentRow] + areaUnderNormalCurve * standardDeviation[currentRow] / squareRootOfN; // We calculate and store the upper mean intervals that were requested.
	}
	
	return;
}

