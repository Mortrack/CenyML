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
#include "CenyMLfeatureScaling.h"


/**
* The "getMinMaxNormalization()" function is used to do two tasks.
* The first is to obtain the parameters required for the min max
* normalization method and to return them when this function
* concludes its work. These parameters are the minimum and the
* maximum values contain with respect to each of the rows that
* are stored in the argument pointer variable "inputMatrix".
* The second is to use these parameters to obtain apply the feature
* scaling method known as the min max normalization method, on the
* data contained in the argument pointer variable "inputMatrix".
* Finally, its result will be stored in the argument pointer
* variable "inputMatrix_dot".
* 
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   input matrix, from which the min
*							   max normalization method will be
*							   calculated.
*
* @param int n - This argument will represent the total number
*				 of rows that the "inputMatrix" variable argument
*				 will have.
*
* @param int m - This argument will represent the total number
*				 of columns that the "inputMatrix" variable argument
*				 will have.
*
* @param double *minMax - This argument will contain the pointer to a
*						  memory allocated variable in which we will
*						  store the individual minimum and maximum
*						  values that were identified from the
*						  argument pointer variable "inputMatrix".
*						  In the "minMax" pointer variable, the
*						  minimum values will be stored in the first
*						  "m" values and the maximum values will be
*						  stored in the last "m" values. In addition,
*						  note that the values will be stored in
*						  ascending order with respect to the column
*						  arrangement. In other words, from the first
*						  column up to the last one. Finally, IT IS
*						  INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*						  BEFORE CALLING THIS FUNCTION.
*
* @param double *inputMatrix_dot - This argument will contain the
* 								   pointer to a memory allocated
*								   variable in which we will store
*								   the values of the argument pointer
*								   variable "inputMatrix" but with the
*								   min max normalization method
*								   applied to it. IT IS INDISPENSABLE
*								   THAT THIS VARIABLE IS ALLOCATED
*								   BEFORE CALLING THIS FUNCTION.
*
* NOTE: RESULTING VALUES IN WHICH THE MIN MAX NORMALIZATION METHOD WAS
*	    APPLIED, IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix_dot". ON THE OTHER HAND, RESULTING VALUES OF THE
*		IDENTIFIED MINIMUM AND MAXIMUM VALUES FOR EACH COLUMN OF 
*		"inputMatrix" ARE STORED IN THE MEMORY ALLOCATED POINTER
*		VARIABLE "minMax".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 07, 2021
* LAST UPDATE: N/A
*/
void getMinMaxNormalization(double *inputMatrix, int n, int m, double *minMax, double *inputMatrix_dot) {
	// We declare and innitialize the local variable to be used to store the maximum values contained in each row of "inputMatrix". Note that the minimum values will be stored directly in the argument pointer variable "minMax".
	double max[m];
	for (int currentColumn = 0; currentColumn < m; currentColumn++) { // We innitialize each of the values of the "min" and "max" variables with the first value of each column of "inputMatrix".
		minMax[currentColumn] = inputMatrix[currentColumn];
		max[currentColumn] = inputMatrix[currentColumn];
	}
	
	// We obtain the individual minimum and maximum values for each column of the argument variable "inputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			if (minMax[currentColumn] > inputMatrix[currentColumn + currentRow*m]) {
				minMax[currentColumn] = inputMatrix[currentColumn + currentRow*m];
			}
			if (max[currentColumn] < inputMatrix[currentColumn + currentRow*m]) {
				max[currentColumn] = inputMatrix[currentColumn + currentRow*m];
			}
		}
	}
	
	// In order to increase performance, we store the difference of "max" and "min", for each column, in the variable "maxMinuxMin" as it is repeated in the mathematical equation of the min max normalization.
	double maxMinuxMin[m];
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		maxMinuxMin[currentColumn] = max[currentColumn] - minMax[currentColumn];
	}
	
	// We obtain the min max normalization for each value contained in "inputMatrix" and store it in "inputMatrix_dot".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			inputMatrix_dot[currentColumn + currentRow*m] = (inputMatrix[currentColumn + currentRow*m] - minMax[currentColumn])/maxMinuxMin[currentColumn];
		}
	}
	
	// We add all the values contained in the "max" variable into the last "m" values of the "min" variable in order to return the values of both variables.
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		minMax[currentColumn + m] = max[currentColumn];
	}
}


/**
* The "getReverseMinMaxNormalization()" function is used to 
* obtain the reverse of the min max normalization method with
* respect to the data contained in the argument pointer variable
* "inputMatrix_dot". The parameters of such method that are
* taken into account will be the ones defined in the argument
* variable "minMax". Finally, the result will be stored in the
* argument pointer variable "inputMatrix".
* 
* @param double *inputMatrix_dot - This argument will contain the
*							   	   pointer to a memory allocated
*							       matrix, from which the reverse
*								   of the min max normalization
*								   method will be calculated.
*
* @param int n - This argument will represent the total number
*				 of rows that the "inputMatrix" variable argument
*				 will have.
*
* @param int m - This argument will represent the total number
*				 of columns that the "inputMatrix" variable argument
*				 will have.
*
* @param double minMax[] - This argument will contain the vector
*						   possessing the data of the minimum and
*						   maximum values that the argument pointer
*						   variable "inputMatrix" should have. The
*						   minimum values should be defined in the
*						   first "m" values and the maximum values
*						   on the last "m" values of the "minMax"
*						   variable.
*
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   matrix in which all the values of the
*							   reverse of the min max normalization
*							   method will be stored. IT IS
*							   INDISPENSABLE THAT THIS VARIABLE IS
*							   ALLOCATED BEFORE CALLING THIS FUNCTION.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 07, 2021
* LAST UPDATE: N/A
*/
void getReverseMinMaxNormalization(double *inputMatrix_dot, int n, int m, double *minMax, double *inputMatrix) {
	// We declare and innitialize the local variables to be used to store the minimum and the maximum values contained in the argument variable "minMax".
	double min[m]; // Variable used to store the minimum values identified for the argument pointer variable "inputMatrix".
	double maxMinusMin[m]; // Variable used to store the difference between the maximum and minimum values identified for the argument pointer variable "inputMatrix".
	for (int currentColumn = 0; currentColumn < m; currentColumn++) { // We innitialize each of the values of the "min" and "max" variables.
		min[currentColumn] = minMax[currentColumn];
		maxMinusMin[currentColumn] = minMax[currentColumn + m] - minMax[currentColumn];
	}
	
	// We obtain the reverse min max normalization for each value contained in "inputMatrix_dot" and store it in "inputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			inputMatrix[currentColumn + currentRow*m] = inputMatrix_dot[currentColumn + currentRow*m] * maxMinusMin[currentColumn] + min[currentColumn];
		}
	}
}

