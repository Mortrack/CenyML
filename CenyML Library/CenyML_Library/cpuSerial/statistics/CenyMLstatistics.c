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
#include <stdlib.h>


/**
* The "getMean()" function is used to calculate the mean of all the
* columns of the input matrix "X" and is returned as a vector (mean).
* 
* @param double* X - This argument will contain the pointer to a
*					 memory allocated input matrix of "X", from
*					 which the mean will be calculated.
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
* CREATEION DATE: SEPTEMBER 23, 2021
* LAST UPDATE: OCTOBER 06, 2021
*/
void getMean(double*  matrix, int n, int m, double* mean) {
	// We declare the local variables to be used.
	int currentRow;
    int currentColumn;
    
	// We obtain the mean for each of the columns of the matrix "X".
    for (currentRow = 0; currentRow < n; currentRow++) {
    	for (currentColumn = 0; currentColumn < m; currentColumn++) {
    		mean[currentColumn] += matrix[(currentColumn + currentRow * m)];
		}
	}
	for (currentColumn = 0; currentColumn < m; currentColumn++) {
		mean[currentColumn] = mean[currentColumn]/n;
	}
	
}

