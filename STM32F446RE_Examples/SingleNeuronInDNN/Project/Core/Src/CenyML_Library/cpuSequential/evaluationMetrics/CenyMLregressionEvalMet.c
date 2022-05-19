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
#include "CenyMLregressionEvalMet.h"


/**
* The "getMeanSquaredError()" function is used to apply a regression
* evaluation metric known as the mean squared error. Such method will
* be applied with respect to the argument pointer variables
* "realOutputMatrix" and "predictedOutputMatrix". Then, its result
* will be stored in the argument pointer variable "MSE".
* 
* @param double *realOutputMatrix - This argument will contain the
*							   		pointer to a memory allocated
*								    output matrix, representing
*									the real data of the system
*									under study. This variable will
*									be used as a reference to
*									compare and apply the mean
*									square metric with respect to
*									the argument pointer variable
*								    "predictedOutputMatrix". THIS
*								    VARIABLE SHOULD BE ALLOCATED
*									AND INNITIALIZED BEFORE CALLING
*									THIS FUNCTION WITH A SIZE OF "n"
*									TIMES "p" 'DOUBLE' MEMORY SPACES.
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
*					   ALLOCATED AND INNITIALIZED WITH ZERO BEFORE
*					   CALLING THIS FUNCTION WITH A SIZE OF "1"
*					   'DOUBLE' MEMORY SPACES where the result will
*					   be stored.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "MSE".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 09, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getMeanSquaredError(double *realOutputMatrix, double *predictedOutputMatrix, int n, int m, int degreesOfFreedom, double *MSE) {
	// We calculate the mean squared error between the argument pointer variables "realOutputMatrix" and "predictedOutputMatrix".
	double squareThisValue; // Variable used to store the value that wants to be squared, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		squareThisValue = realOutputMatrix[currentRow] - predictedOutputMatrix[currentRow];
		MSE[0] += (squareThisValue * squareThisValue);
	}
	MSE[0] = MSE[0] / (n - m - degreesOfFreedom);
	
	return;
}


/**
* The "getCoefficientOfDetermination()" function is used to apply
* a regression evaluation metric known as the coefficient of
* determination. Such method will be applied with respect to the
* argument pointer variables "realOutputMatrix" and
* "predictedOutputMatrix". Then, its result will be stored in the
* argument pointer variable "Rsquared".
* 
* @param double *realOutputMatrix - This argument will contain the
*							   		pointer to a memory allocated
*								    output matrix, representing
*									the real data of the system
*									under study. This variable will
*									be used as a reference to
*									compare and apply the coefficient
*									of determination metric with
*									respect to the argument pointer
*									variable "predictedOutputMatrix".
*									THIS VARIABLE SHOULD BE ALLOCATED
*									AND INNITIALIZED BEFORE CALLING
*									THIS FUNCTION WITH A SIZE OF "n"
*									TIMES "p" 'DOUBLE' MEMORY SPACES.
*
* @param double *predictedOutputMatrix - This argument will contain
*										 the pointer to a memory
*										 allocated output matrix,
*										 representing the predicted
*							   			 data of the system under
*										 study. The data contained
*										 in this variable will be
*										 evaluated with the
*										 coefficient of determination
*										 metric. THIS VARIABLE SHOULD
*										 BE ALLOCATED AND INNITIALIZED
*										 BEFORE CALLING THIS FUNCTION
*										 WITH A SIZE OF "n" TIMES "p"
*										 'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param double *Rsquared - This argument will contain the pointer to a
*					   		memory allocated variable in which we will
*							store the resulting metric evaluation
*							obtained after having applied the
*							coefficient of determination metric between
*							the argument pointer variables
*					   		"realOutputMatrix" and
*							"predictedOutputMatrix". IT IS INDISPENSABLE
*							THAT THIS VARIABLE IS ALLOCATED AND
*							INNITIALIZED WITH ZERO BEFORE CALLING THIS
*							FUNCTION WITH A SIZE OF "1" 'DOUBLE' MEMORY
*							SPACES where the result will be stored.
*
* NOTE: RESULTS ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "Rsquared".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 09, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getCoefficientOfDetermination(double *realOutputMatrix, double *predictedOutputMatrix, int n, double *Rsquared) {
	// We obtain the sums required for the means to be calculated and the SSE values for each of the columns of the input matrix.
	double squareThisValue; // Variable used to store the value that wants to be squared, for performance purposes.
	double mean_realOutputMatrix = 0; // This variable is used to store the means of all the outputs of the argument pointer variable "realOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		// We make the required calculations to obtain the SSE values.
		squareThisValue = realOutputMatrix[currentRow] - predictedOutputMatrix[currentRow];
		Rsquared[0] += (squareThisValue * squareThisValue); // We temporarly store the SSE values in the argument pointer variable "Rsquared", for performance purposes.
		
		// We make the required calculations to obtain the sums required for the calculation of the means.
    		mean_realOutputMatrix += realOutputMatrix[currentRow];
	}
	// We apply the final operations required to complete the calculation of the means.
	mean_realOutputMatrix = mean_realOutputMatrix/n;
	
	// We obtain the SST values that will be required to make the calculation of the coefficient of determination.
	double SST = 0; // This variable is used to store the SST values for all the outputs of the argument pointer variable "realOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		// We make the required calculations to obtain the SST values.
		squareThisValue = realOutputMatrix[currentRow] - mean_realOutputMatrix;
		SST += (squareThisValue * squareThisValue); // We temporarly store the SSE values in the argument pointer variable "R", for performance purposes.
	}
	
	// Finally, we calculate the coefficient of determination and store its results in the pointer variable "Rsquared".
	Rsquared[0] = 1 - (Rsquared[0]/SST);
	
	return;
}


/**
* The "getAdjustedCoefficientOfDetermination()" function is used to
* apply a regression evaluation metric known as the adjusted
* coefficient of determination. Such method will be applied with
* respect to the argument pointer variables "realOutputMatrix" and
* "predictedOutputMatrix". Then, its result will be stored in the
* argument pointer variable "adjustedRsquared".
* 
* @param double *realOutputMatrix - This argument will contain the
*							   		pointer to a memory allocated
*								    output matrix, representing
*									the real data of the system
*									under study. This variable will
*									be used as a reference to
*									compare and apply the adjusted
*									coefficient of determination
*									metric with respect to the
*									argument pointer variable
*									"predictedOutputMatrix". THIS
*									VARIABLE SHOULD BE ALLOCATED AND
*									INNITIALIZED BEFORE CALLING THIS
*									FUNCTION WITH A SIZE OF "n" TIMES
*									"p" 'DOUBLE' MEMORY SPACES.
*
* @param double *predictedOutputMatrix - This argument will contain
*										 the pointer to a memory
*										 allocated output matrix,
*										 representing the predicted
*							   			 data of the system under
*										 study. The data contained
*										 in this variable will be
*										 evaluated with the adjusted
*										 coefficient of determination
*										 metric. THIS VARIABLE SHOULD
*										 BE ALLOCATED AND INNITIALIZED
*										 BEFORE CALLING THIS FUNCTION
*										 WITH A SIZE OF "n" TIMES "p"
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
* @param double *adjustedRsquared - This argument will contain the
*									pointer to a memory allocated
*									variable in which we will store the
*									resulting metric evaluation obtained
*									after having applied the adjusted
*									coefficient of determination metric
*									between the argument pointer variables
*					   				"realOutputMatrix" and
*									"predictedOutputMatrix". IT IS
*									INDISPENSABLE
*									THAT THIS VARIABLE IS ALLOCATED AND
*									INNITIALIZED WITH ZERO BEFORE CALLING
*									THIS FUNCTION WITH A SIZE OF "1"
*									'DOUBLE' MEMORY SPACES where the
*									result will be stored.
*
* NOTE: RESULTS ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "adjustedRsquared".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 12, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getAdjustedCoefficientOfDetermination(double *realOutputMatrix, double *predictedOutputMatrix, int n, int m, int degreesOfFreedom, double *adjustedRsquared) {
	// We obtain the sums required for the means to be calculated and the SSE values for each of the columns of the input matrix.
	double squareThisValue; // Variable used to store the value that wants to be squared, for performance purposes.
	double mean_realOutputMatrix = 0; // This variable is used to store the means of all the outputs of the argument pointer variable "realOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		// We make the required calculations to obtain the SSE values.
		squareThisValue = realOutputMatrix[currentRow] - predictedOutputMatrix[currentRow];
		adjustedRsquared[0] += (squareThisValue * squareThisValue); // We temporarly store the SSE values in the argument pointer variable "adjustedRsquared", for performance purposes.
		
		// We make the required calculations to obtain the sums required for the calculation of the means.
    		mean_realOutputMatrix += realOutputMatrix[currentRow];
	}
	// We apply the final operations required to complete the calculation of the means.
	mean_realOutputMatrix = mean_realOutputMatrix/n;
	
	// We obtain the SST values that will be required to make the calculation of the adjusted coefficient of determination.
	double SST = 0; // This variable is used to store the SST values for all the outputs of the argument pointer variable "realOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		// We make the required calculations to obtain the SST values.
		squareThisValue = realOutputMatrix[currentRow] - mean_realOutputMatrix;
		SST += (squareThisValue * squareThisValue); // We temporarly store the SSE values in the argument pointer variable "R", for performance purposes.
	}
	
	// Finally, we calculate the adjusted coefficient of determination and store its results in the pointer variable "adjustedRsquared".
	adjustedRsquared[0] = 1 - ( (adjustedRsquared[0]/(n-m-degreesOfFreedom))/(SST/(n-degreesOfFreedom)) );
	
	return;
}

