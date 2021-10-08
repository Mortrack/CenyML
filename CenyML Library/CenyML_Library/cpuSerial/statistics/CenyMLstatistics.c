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


/**
* The "getMean()" function is used to calculate the mean of all the
* columns of the input matrix "X" and is returned as a vector (mean).
* 
* @param double* X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which
*					 the mean will be calculated.
*
* @param int n - This argument will represent the total number
*				 of rows that the "data" variable argument will
*				 have.
*
* @param int m - This argument will represent the total number
*				 of columns that the "data" variable argument
*				 will have.
*
* @param double* mean - This argument will contain the pointer to
*						a memory allocated variable in which we
*						will store the mean for each column
*						contained in the argument variable "X". IT
*						IS INDISPENSABLE THAT THIS VARIABLE IS
*						INNITIALIZED BEFORE CALLING THIS FUNCTION.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "mean".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: SEPTEMBER 23, 2021
* LAST UPDATE: OCTOBER 06, 2021
*/
void getMean(double* matrix, int n, int m, double* mean) {
	// We obtain the mean for each of the columns of the matrix "X".
    for (int currentRow = 0; currentRow < n; currentRow++) {
    	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
    		mean[currentColumn] += matrix[(currentColumn + currentRow*m)];
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
*				 of rows that the "inputMatrix" variable argument
*				 will have.
*
* @param int m - This argument will represent the total number
*				 of columns that the "inputMatrix" variable
*				 argument will have.
*
* @param double* inputMatrix - This argument will contain the
*							   pointer to a memory allocated input
*							   matrix, from which a certain sort
*							   method will be applied.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: OCTOBER 07, 2021
* LAST UPDATE: N/A
*/
void getSort(char desiredSortMethod[], int n, int m, double* inputMatrix) {
	// If the implementer requested the "quick sort" method, then apply it through the following code.
	if (strcmp(desiredSortMethod, "quicksort") == 0) {
		// We allocate the needed memory for the vector containing the rows of a certain column from the "inputMatrix".
		double* rowsOfCurrentColumn = (double*)malloc(n*sizeof(double));
		
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
* @param double* inputVector - This argument will contain the
*							   pointer to a memory allocated input
*							   vector, from which the "quick sort"
*							   method will be applied.
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
* LAST UPDATE: N/A
*/
static void applyQuickSort(int minIndexLimit, int maxIndexLimit, double* inputVector) {
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

