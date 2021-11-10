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
#include "CenyMLregressionEvalMet.h"


/**
* The "getMeanSquaredError()" function is used to apply a regression
* evaluation metric known as the mean squared error. Such method will
* be applied with respect to the argument pointer variables
* "realOutputMatrix" and "predictedOutputMatrix". Then, its result
* will be stored in the argument pointer variable "MSE".
* 
* @param double *realOutputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   output matrix, representing the real
*							   data of the system under study. This
*							   variable will be used as a reference
*							   to compare and apply the mean square
*							   metric with respect to the argument
*							   pointer variable
*							   "predictedOutputMatrix". THIS
*							   VARIABLE SHOULD BE ALLOCATED AND
*							   INNITIALIZED BEFORE CALLING THIS
*							   FUNCTION WITH A SIZE OF "n" TIMES "p"
*							   'DOUBLE' MEMORY SPACES.
*
* @param double *predictedOutputMatrix - This argument will contain
*										 the pointer to a memory
*										 allocated output matrix,
*										 representing the predicted
*							   			 data of the system under
*										 study. The data contained
*										 in this variable will be
*										 evaluated with the mean
*										 squared error metric. THIS
*							   			 VARIABLE SHOULD BE ALLOCATED
*										 AND INNITIALIZED BEFORE
*										 CALLING THIS FUNCTION WITH A
*										 SIZE OF "n" TIMES "p"
*										 'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of 
*				 features (independent variables) that the input matrix
*				 has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix, containing
*				 the real/predicted results of the system under study.
*
* @param int degreesOfFreedom - This argument will represent the desired
*								value for the degrees of freedom to be
*								applied when calculating the variance.
*
* @param double *MSE - This argument will contain the pointer to a
*					   memory allocated variable in which we will store
*					   the resulting metric evaluation obtained after
*					   having applied the mean squared error metric
*					   between the argument pointer variables
*					   "realOutputMatrix" and "predictedOutputMatrix".
*					   IT IS INDISPENSABLE THAT THIS VARIABLE IS
*					   ALLOCATED BEFORE CALLING THIS FUNCTION WITH A
*					   SIZE OF "p" 'DOUBLE' MEMORY SPACES, WHERE "p"
*					   STANDS FOR THE NUMBER OF OUTPUTS THAT THE SYSTEM
*					   UNDER STUDY HAS. Note that the results will be
*					   stored in ascending order with respect to the
*					   outputs of the system under study. In other words,
*					   from the first output in index "0" up to the last
*					   output in index "p-1".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "MSE".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 09, 2021
* LAST UPDATE: N/A
*/
void getMeanSquaredError(double *realOutputMatrix, double *predictedOutputMatrix, int n, int m, int p, int degreesOfFreedom, double *MSE) {
	// We calculate the mean squared error between the argument pointer variables "realOutputMatrix" and "predictedOutputMatrix".
	int currentRowTimesP; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	double squareThisValue; // Variable used to store the value that wants to be squared, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesP = currentRow*p;
		for (int currentOutput=0; currentOutput<p; currentOutput++) {
			currentRowAndColumn = currentOutput + currentRowTimesP;
			squareThisValue = realOutputMatrix[currentRowAndColumn] - predictedOutputMatrix[currentRowAndColumn];
			MSE[currentOutput] += squareThisValue * squareThisValue;
		}
	}
	for (int currentOutput=0; currentOutput<p; currentOutput++) {
		MSE[currentOutput] = MSE[currentOutput] / (n - m - degreesOfFreedom);
	}
}

