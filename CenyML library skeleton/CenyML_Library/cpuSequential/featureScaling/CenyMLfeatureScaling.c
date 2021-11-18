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
// IMPORTANT NOTE: This library uses the math.h library and therefore,
//				   remember to use the "-lm" flag when compiling it.


/**
* The "getMinMaxNormalization()" function is used to do two tasks.
* The first is to obtain the parameters required for the min max
* normalization method and to return them when this function
* concludes its work. These parameters are the minimum and the
* maximum values contain with respect to each of the rows that
* are stored in the argument pointer variable "inputMatrix".
* The second is to use these parameters to apply the feature
* scaling method known as the min max normalization method, on the
* data contained in the argument pointer variable "inputMatrix".
* Finally, its result will be stored in the argument pointer
* variable "inputMatrix_dot".
* 
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   input matrix, from which the min
*							   max normalization method will be
*							   calculated. THIS VARIABLE SHOULD
*							   BE ALLOCATED AND INNITIALIZED
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
*						  BEFORE CALLING THIS FUNCTION WITH A SIZE OF 
*						  TWO TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* @param double *inputMatrix_dot - This argument will contain the
* 								   pointer to a memory allocated
*								   variable in which we will store
*								   the values of the argument pointer
*								   variable "inputMatrix" but with the
*								   min max normalization method
*								   applied to it. IT IS INDISPENSABLE
*								   THAT THIS VARIABLE IS ALLOCATED
*								   BEFORE CALLING THIS FUNCTION WITH A
*								   SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*								   SPACES.
*
* NOTE: RESULTING VALUES IN WHICH THE MIN MAX NORMALIZATION METHOD WAS
*	    APPLIED, ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix_dot". ON THE OTHER HAND, RESULTING VALUES OF THE
*		IDENTIFIED MINIMUM AND MAXIMUM VALUES FOR EACH COLUMN OF 
*		"inputMatrix" ARE STORED IN THE MEMORY ALLOCATED POINTER
*		VARIABLE "minMax".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 07, 2021
* LAST UPDATE: NOVEMBER 09, 2021
*/
void getMinMaxNormalization(double *inputMatrix, int n, int m, double *minMax, double *inputMatrix_dot) {
	// We declare and innitialize the local variable to be used to store the maximum values contained in each row of "inputMatrix". Note that the minimum values will be stored directly in the argument pointer variable "minMax".
	double max[m];
	for (int currentColumn = 0; currentColumn < m; currentColumn++) { // We innitialize each of the values of the "min" and "max" variables with the first value of each column of "inputMatrix".
		minMax[currentColumn] = inputMatrix[currentColumn];
		max[currentColumn] = inputMatrix[currentColumn];
	}
	
	// We obtain the individual minimum and maximum values for each column of the argument variable "inputMatrix".
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	for (int currentRow = 1; currentRow < n; currentRow++) { // We start from the second row because we already went through the first one in the previous for-loop.
		currentRowTimesM = currentRow*m;
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesM;
			if (minMax[currentColumn] > inputMatrix[currentRowAndColumn]) {
				minMax[currentColumn] = inputMatrix[currentRowAndColumn];
			}
			if (max[currentColumn] < inputMatrix[currentRowAndColumn]) {
				max[currentColumn] = inputMatrix[currentRowAndColumn];
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
		currentRowTimesM = currentRow*m;
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesM;
			inputMatrix_dot[currentRowAndColumn] = (inputMatrix[currentRowAndColumn] - minMax[currentColumn])/maxMinuxMin[currentColumn];
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
*								   method will be calculated.THIS
*								   VARIABLE SHOULD BE ALLOCATED
*								   AND INNITIALIZED BEFORE CALLING
*								   THIS FUNCTION WITH A SIZE OF "n"
*								   TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number
*				 of samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number
*				 of features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param double *minMax - This argument will contain the vector
*						  possessing the data of the minimum and
*						  maximum values that the argument pointer
*						  variable "inputMatrix" should have. The
*						  minimum values should be defined in the
*						  first "m" values and the maximum values
*						  on the last "m" values of the "minMax"
*						  variable. THIS VARIABLE SHOULD BE
*						  ALLOCATED AND INNITIALIZED BEFORE CALLING
*						  THIS FUNCTION WITH A SIZE OF TWO TIMES "m"
*						  'DOUBLE' MEMORY SPACES.
*
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   matrix in which all the values of the
*							   reverse of the min max normalization
*							   method will be stored. IT IS
*							   INDISPENSABLE THAT THIS VARIABLE IS
*							   ALLOCATED BEFORE CALLING THIS FUNCTION
*							   WITH A VARIABLE SIZE OF "m" TIMES "n"
*							   'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 07, 2021
* LAST UPDATE: NOVEMBER 09, 2021
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
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesM = currentRow*m;
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesM;
			inputMatrix[currentRowAndColumn] = inputMatrix_dot[currentRowAndColumn] * maxMinusMin[currentColumn] + min[currentColumn];
		}
	}
}


/**
* The "getL2Normalization()" function is used to do two tasks.
* The first is to obtain the parameter required for the L2
* normalization and to return it when this function concludes its
* work. This parameter is the magnitude with respect to each
* column of the argument pointer variable "inputMatrix". The
* second is to use this parameter to apply the feature scaling
* method known as the L2 normalization method, on the data
* contained in the argument pointer variable "inputMatrix".
* Finally, its result will be stored in the argument pointer
* variable "inputMatrix_dot".
* 
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   input matrix, from which the L2
*							   normalization method will be
*							   calculated. THIS VARIABLE SHOULD
*							   BE ALLOCATED AND INNITIALIZED
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
* @param double *magnitude - This argument will contain the pointer
*							 to a memory allocated variable in which
*	 						 we will store the individual magintude
*							 values that were identified from the
*						  	 argument pointer variable "inputMatrix"
*							 with respect to each of its columns.
*						  	 In the "magnitude" pointer variable, the
*							 values will be stored in ascending order
*							 with respect to the column arrangement. In
*							 other words, from the first column up to
*							 the last one. Finally, IT IS INDISPENSABLE
*							 THAT THIS VARIABLE IS ALLOCATED AND
*							 INITIALIZED WITH ZEROS BEFORE CALLING THIS
*							 FUNCTION WITH A SIZE OF "m" 'DOUBLE'
*							 MEMORY SPACES.
*
* @param double *inputMatrix_dot - This argument will contain the
* 								   pointer to a memory allocated
*								   variable in which we will store
*								   the values of the argument pointer
*								   variable "inputMatrix" but with the
*								   L2 normalization method applied to
*								   it. IT IS INDISPENSABLE THAT THIS
*								   VARIABLE IS ALLOCATED BEFORE CALLING
*								   THIS FUNCTION WITH A SIZE OF "n"
*								   TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULTING VALUES IN WHICH THE L2 NORMALIZATION METHOD WAS
*	    APPLIED, ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix_dot". ON THE OTHER HAND, RESULTING VALUES OF THE
*		IDENTIFIED MAGNITUDE VALUES FOR EACH COLUMN OF "inputMatrix"
*		ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "magnitude".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 08, 2021
* LAST UPDATE: NOVEMBER 18, 2021
*/
void getL2Normalization(double *inputMatrix, int n, int m, double *magnitude, double *inputMatrix_dot) {
	// In order to incrase performance, we will first calculate the magnitude with respect to each column of the argument pointer variable "inputMatrix".
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesM = currentRow*m;
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesM;
			magnitude[currentColumn] += inputMatrix[currentRowAndColumn] * inputMatrix[currentRowAndColumn];
		}
	}
	for (int currentColumn = 0; currentColumn < m; currentColumn++) { // To conclude, we apply the square root to the resulting sums that were made.
		magnitude[currentColumn] = sqrt(magnitude[currentColumn]);
	}
	
	// We obtain the L2 normalization for each value contained in "inputMatrix" and store it in "inputMatrix_dot".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesM = currentRow*m;
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesM;
			inputMatrix_dot[currentRowAndColumn] = inputMatrix[currentRowAndColumn] / magnitude[currentColumn];
		}
	}
}


/**
* The "getReverseL2Normalization()" function is used to obtain the
* reverse of the L2 normalization method with respect to the data
* contained in the argument pointer variable "inputMatrix_dot".
* The parameter of such method that is taken into account will be
* the one defined in the argument variable "magnitude". Finally,
* the result will be stored in the argument pointer variable
* "inputMatrix".
* 
* @param double *inputMatrix_dot - This argument will contain the
*							   	   pointer to a memory allocated
*							       matrix, from which the reverse
*								   of the L2 normalization method
*								   will be calculated.THIS VARIABLE
*								   SHOULD BE ALLOCATED AND
*								   INNITIALIZED BEFORE CALLING THIS
*								   FUNCTION WITH A SIZE OF "n"
*								   TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number
*				 of samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number
*				 of features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param double *magnitude - This argument will contain the pointer
*							 to a memory allocated variable in which
*	 						 the individual magnitude values of each of
*							 the columns of the argument pointer
*							 variable "inputMatrix" should be defined
*							 previously to calling this function. In
*							 the "magnitude" pointer variable, the
*							 values should be stored in ascending order
*							 with respect to the column arrangement. In
*							 other words, from the first column up to
*							 the last one. Finally, IT IS INDISPENSABLE
*							 THAT THIS VARIABLE IS ALLOCATED AND 
*							 INNITIALIZED BEFORE CALLING THIS FUNCTION
*							 WITH A SIZE OF "m" 'DOUBLE' MEMORY SPACES.
*
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   matrix in which all the values of the
*							   reverse of the L2 normalization method
*							   will be stored. IT IS INDISPENSABLE THAT
*							   THIS VARIABLE IS ALLOCATED BEFORE CALLING
*							   THIS FUNCTION WITH A VARIABLE SIZE OF "m"
*							   TIMES "n" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 08, 2021
* LAST UPDATE: NOVEMBER 09, 2021
*/
void getReverseL2Normalization(double *inputMatrix_dot, int n, int m, double *magnitude, double *inputMatrix) {
	// We obtain the reverse L2 normalization for each value contained in "inputMatrix_dot" and store it in "inputMatrix".
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesM = currentRow*m;
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesM;
			inputMatrix[currentRowAndColumn] = inputMatrix_dot[currentRowAndColumn] * magnitude[currentColumn];
		}
	}
}


/**
* The "getZscoreNormalization()" function is used to do two tasks.
* The first is to obtain the parameters required for the Z score
* normalization and to return it when this function concludes its
* work. These parameters are the mean and standard deviation with
* respect to each column of the argument pointer variable
* "inputMatrix". The second is to use these parameters to apply
* the feature scaling method known as the Z score normalization
* method, on the data contained in the argument pointer variable
* "inputMatrix". Finally, its result will be stored in the
* argument pointer variable "inputMatrix_dot".
* 
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   input matrix, from which the Z
*							   score normalization method will be
*							   calculated. THIS VARIABLE SHOULD
*							   BE ALLOCATED AND INNITIALIZED
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
*								applied when calculating the standard
*								deviation.
*
* @param double *meanAndstdDev - This argument will contain the pointer
*							 	 to a memory allocated variable in
*								 which we will store the individual
*								 mean and standard deviation values
*								 that were identified from the argument
*								 pointer variable "inputMatrix" with
*								 respect to each of its columns. In the
*								 "meanAndstdDev" pointer variable, the
*								 mean and standard deviation for each
*								 column will be stored in a different
*								 row in ascending order, where the first
*								 row of "meanAndstdDev" will have the
*							     parameters for the first column of the
*								 pointer variable "inputMatrix" and the
*								 last row of "meanAndstdDev" will have
*								 the parameters for the last column of
*								 the pointer variable "inputMatrix".
*								 Finally, IT IS INDISPENSABLE THAT THIS
*								 VARIABLE IS ALLOCATED AND INITIALIZED
*								 WITH ZEROS BEFORE CALLING THIS
*								 FUNCTION WITH A SIZE OF TWO TIMES "m"
*								 'DOUBLE' MEMORY SPACES.
*
* @param double *inputMatrix_dot - This argument will contain the
* 								   pointer to a memory allocated
*								   variable in which we will store
*								   the values of the argument pointer
*								   variable "inputMatrix" but with the
*								   Z score normalization method applied
*								   to it. IT IS INDISPENSABLE THAT THIS
*								   VARIABLE IS ALLOCATED BEFORE CALLING
*								   THIS FUNCTION WITH A SIZE OF "n"
*								   TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULTING VALUES IN WHICH THE Z SCORE NORMALIZATION METHOD WAS
*	    APPLIED, ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix_dot". ON THE OTHER HAND, RESULTING VALUES OF THE
*		IDENTIFIED MEAN AND STANDARD DEVIATION VALUES FOR EACH COLUMN
*		OF "inputMatrix" ARE STORED IN THE MEMORY ALLOCATED POINTER
*		VARIABLE "meanAndstdDev" (For more details on "meanAndstdDev",
*		see its description).
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 08, 2021
* LAST UPDATE: NOVEMBER 18, 2021
*/
void getZscoreNormalization(double *inputMatrix, int n, int m, int degreesOfFreedom, double *meanAndstdDev, double *inputMatrix_dot) {
	// We obtain the mean for each of the columns of the input matrix.
    int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesM = currentRow*m;
    	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
    		meanAndstdDev[2*currentColumn] += inputMatrix[currentColumn + currentRowTimesM];
		}
	}
	int twoTimesCurrentColumn; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		twoTimesCurrentColumn = 2*currentColumn;
		meanAndstdDev[twoTimesCurrentColumn] = meanAndstdDev[twoTimesCurrentColumn]/n;
	}
	
	// We calculate the variance and then obtain its square root to get the standard deviation for each of the columns of the input matrix.
    double squareThisValue; // Variable used to store the value that wants to be squared.
    for (int currentRow = 0; currentRow < n; currentRow++) {
    	currentRowTimesM = currentRow*m;
    	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
    		twoTimesCurrentColumn = 2*currentColumn;
    		squareThisValue = inputMatrix[(currentColumn + currentRowTimesM)] - meanAndstdDev[twoTimesCurrentColumn];
    		meanAndstdDev[twoTimesCurrentColumn + 1] += squareThisValue*squareThisValue;
		}
	}
	degreesOfFreedom = n-degreesOfFreedom; // We apply the requested degrees of freedom.
	for (int currentColumn = 0; currentColumn < m; currentColumn++) {
		twoTimesCurrentColumn = 2*currentColumn;
		meanAndstdDev[twoTimesCurrentColumn + 1] = sqrt(meanAndstdDev[twoTimesCurrentColumn + 1]/degreesOfFreedom);
	}
	
	// We obtain the Z score normalization for each value contained in "inputMatrix" and store it in "inputMatrix_dot".
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesM = currentRow*m;
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesM;
			twoTimesCurrentColumn = 2*currentColumn;
			inputMatrix_dot[currentRowAndColumn] = (inputMatrix[currentRowAndColumn] - meanAndstdDev[twoTimesCurrentColumn]) / meanAndstdDev[twoTimesCurrentColumn + 1];
		}
	}
}


/**
* The "getReverseZscoreNormalization()" function is used to obtain the
* reverse of the Z score normalization method with respect to the data
* contained in the argument pointer variable "inputMatrix_dot".
* The parameters of such method that are taken into account will be
* the ones defined in the argument variable "meanAndstdDev". Finally, the
* result will be stored in the argument pointer variable
* "inputMatrix".
* 
* @param double *inputMatrix_dot - This argument will contain the
*							   	   pointer to a memory allocated
*							       matrix, from which the reverse
*								   of the Z score normalization method
*								   will be calculated.THIS VARIABLE
*								   SHOULD BE ALLOCATED AND
*								   INNITIALIZED BEFORE CALLING THIS
*								   FUNCTION WITH A SIZE OF "n"
*								   TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number
*				 of samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number
*				 of features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param double *meanAndstdDev - This argument will contain the pointer
*							 	 to a memory allocated variable in
*								 which the individual mean and standard
*								 deviation values with respect to each
*								 of the columns of "inputMatrix" should
*								 be assigned previously to calling this
*								 function. In the "meanAndstdDev"
*								 pointer variable, the mean and standard
*								 deviation for each column should be
*								 stored in a different row in ascending
*								 order, where the first row of
*								 "meanAndstdDev" should have the parameters
*								 for the first column of the pointer
*								 variable "inputMatrix" and the last row
*								 of "meanAndstdDev" should have the
*								 parameters for the last column of the
*								 pointer variable "inputMatrix". Finally,
*								 IT IS INDISPENSABLE THAT THIS VARIABLE
*								 IS ALLOCATED AND INNITIALIZED BEFORE
*								 CALLING THIS FUNCTION WITH A SIZE OF TWO
*								 TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* @param double *inputMatrix - This argument will contain the pointer to
*							   a memory allocated matrix in which all the
*							   values of the reverse of the Z score
*							   normalization method will be stored. IT IS
*							   INDISPENSABLE THAT THIS VARIABLE IS
*							   ALLOCATED BEFORE CALLING THIS FUNCTION WITH
*							   A VARIABLE SIZE OF "m" TIMES "n" 'DOUBLE'
*							   MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 08, 2021
* LAST UPDATE: NOVEMBER 09, 2021
*/
void getReverseZscoreNormalization(double *inputMatrix_dot, int n, int m, double *meanAndstdDev, double *inputMatrix) {
	// We obtain the reverse Z score normalization for each value contained in "inputMatrix_dot" and store it in "inputMatrix".
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int twoTimesCurrentColumn; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesM = currentRow*m;
		for (int currentColumn = 0; currentColumn < m; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesM;
			twoTimesCurrentColumn = 2*currentColumn;
			inputMatrix[currentRowAndColumn] = inputMatrix_dot[currentRowAndColumn] * meanAndstdDev[twoTimesCurrentColumn + 1] + meanAndstdDev[twoTimesCurrentColumn];
		}
	}
}

