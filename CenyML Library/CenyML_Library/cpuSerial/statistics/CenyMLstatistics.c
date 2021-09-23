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
* columns of the input matrix "X" and is returned as a vector (B_x_bar).
* 
* @param double* X - This argument will contain the input matrix
*					 values of "X", from which the mean will be
*					 calculated.
*
* @param int n - This argument will represent the total number
*				 of rows that the "data" variable argument will
*				 have.
*
* @param int m - This argument will represent the total number
*				 of columns that the "data" variable argument
*				 will have.
* 
* @return double[m]
* 
* @author Miranda Meza Cesar
* CREATEION DATE: SEPTEMBER 23, 2021
* UPDATE DATE: N/A
*/
double* getMean(double* X, double n, double m) {
	// Allocate the memory required for the struct variable "X", which will contain the input data of the system.
	double totalElementsPerMatrix = m;
	double nBytes = totalElementsPerMatrix * sizeof(double);
	double* B_x_bar = (double*)malloc(nBytes);
	double currentRow;
    double currentColumn;
    // We innitialize the values of "B_x_bar".
	for (currentColumn = 0; currentColumn < m; currentColumn++) {
		B_x_bar[(int)currentColumn] = 0;
	}
	// We obtain the mean for each of the columns of the matrix "X".
    for (currentRow = 0; currentRow < n; currentRow++) {
    	for (currentColumn = 0; currentColumn < m; currentColumn++) {
    		B_x_bar[(int)currentColumn] += X[(int)(currentColumn + currentRow * m)];
		}
	}
	for (currentColumn = 0; currentColumn < m; currentColumn++) {
		B_x_bar[(int)currentColumn] = B_x_bar[(int)currentColumn]/n;
	}
	
	return B_x_bar;
}

