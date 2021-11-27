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
#include "CenyMLclassificationEvalMet.h"
// IMPORTANT NOTE: This library uses the math.h library and therefore,
//				   remember to use the "-lm" flag when compiling it.


/**
* The "getCrossEntropyError()" function is used to apply a
* classification evaluation metric known as the cross entropy error.
* Such method will be applied with respect to the argument pointer
* variables "realOutputMatrix" and "predictedOutputMatrix". Then, its
* result will be stored in the argument pointer variable "NLL".
* 
* @param double *realOutputMatrix - This argument will contain the
*							   		pointer to a memory allocated
*								    output matrix, representing
*									the real data of the system
*									under study. This variable will
*									be used as a reference to
*									compare and apply the cross
*									entropy metric with respect to
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
*										 evaluated with the cross
*										 entropy error metric. THIS
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
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix, containing
*				 the real/predicted results of the system under study.
*
* @param double NLLepsilon - This argument will contain the user defined
*							 epsilon value that will be used to temporarly
*							 store the sum of any "0" value with it and
*							 substract it to any "1" value of any of the
*							 output matrixes ("realOutputMatrix" and/or
*							 "predictedOutputMatrix"). This process is a
*							 strictly needed mathematical operation in
*							 the calculus of the desired error metric.
*							 If not followed, a mathematical error will
*							 be obtained due to the calculation of ln(0).
*							 IMPORTANT NOTE: The results will be
*							 temporarly stored so that the values of the
*							 output matrixes are not modified.
*
* @param double *NLL - This argument will contain the pointer to a
*					   memory allocated variable in which we will store
*					   the resulting metric evaluation obtained after
*					   having applied the cross entropy error metric
*					   between the argument pointer variables
*					   "realOutputMatrix" and "predictedOutputMatrix".
*					   IT IS INDISPENSABLE THAT THIS VARIABLE IS
*					   ALLOCATED AND INNITIALIZED WITH ZERO BEFORE
*					   CALLING THIS FUNCTION WITH A SIZE OF "p" 'DOUBLE'
*					   MEMORY SPACES, WHERE "p" STANDS FOR THE NUMBER OF
*					   OUTPUTS THAT THE SYSTEM UNDER STUDY HAS. Note that
*					   the results will be stored in ascending order with
*					   respect to the outputs of the system under study.
*					   In other words, from the first output in index "0"
*					   up to the last output in index "p-1".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "NLL".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 22, 2021
* LAST UPDATE: N/A
*/
void getCrossEntropyError(double *realOutputMatrix, double *predictedOutputMatrix, int n, int p, double NLLepsilon, double *NLL) {
	// We calculate the cross entropy error between the argument pointer variables "realOutputMatrix" and "predictedOutputMatrix".
	int currentRowTimesP; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	double oneMinusEpsilon = 1 - NLLepsilon; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	double rOM; // This variable is used to store the current value of the real output matrix in which, if the value if zero, it will be replaced with the value of the argument variable "NLLepsilon".
	double pOM; // This variable is used to store the current value of the predicted output matrix in which, if the value if zero, it will be replaced with the value of the argument variable "NLLepsilon".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesP = currentRow*p;
		for (int currentOutput=0; currentOutput<p; currentOutput++) {
			// We temporarly store the sum the user defined epsilon value in "NLLepsilon" to any "0" value and substract it to any "1" value of the real output matrix.
			currentRowAndColumn = currentOutput + currentRowTimesP;
			if (realOutputMatrix[currentRowAndColumn] == 0) {
				rOM = NLLepsilon;
			} else if (realOutputMatrix[currentRowAndColumn] == 1) {
				rOM = oneMinusEpsilon;
			} else {
				rOM = realOutputMatrix[currentRowAndColumn];
			}
			
			// We temporarly store the sum the user defined epsilon value in "NLLepsilon" to any "0" value and substract it to any "1" value of the predicted output matrix.
			if (predictedOutputMatrix[currentRowAndColumn] == 0) {
				pOM = NLLepsilon;
			} else if (predictedOutputMatrix[currentRowAndColumn] == 1) {
				pOM = oneMinusEpsilon;
			} else {
				pOM = predictedOutputMatrix[currentRowAndColumn];
			}
			
			// We calculate the current error and add it into the memory space were it will be stored.
			NLL[currentOutput] = NLL[currentOutput] - rOM*log(pOM) - (1-rOM) * log(1-pOM);
		}
	}
}


/**
* The "getConfusionMatrix()" function is used to calculate and obtain
* the classification evaluation metric known as the confusion matrix.
* Such method will be applied with respect to the argument pointer
* variables "realOutputMatrix" and "predictedOutputMatrix". Then, its
* result will be stored in the argument pointer variable
* "confusionMatrix".
* 
* @param double *realOutputMatrix - This argument will contain the
*							   		pointer to a memory allocated
*								    output matrix, representing
*									the real data of the system
*									under study. This variable will
*									be used as a reference to
*									calculate and obtain the
*									confusion matrix with respect to
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
*										 used to calculate and obtain
*										 the confusion matrix. THIS
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
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix, containing
*				 the real/predicted results of the system under study.
*
* @param double *confusionMatrix - This argument will contain the pointer
*								   to a memory allocated variable in which
*								   we will store the resulting metric
*								   evaluation obtained after having applied
*								   the confusion matrix metric between the
*								   argument pointer variables
*								   "realOutputMatrix" and
*								   "predictedOutputMatrix". IT IS
*								   INDISPENSABLE THAT THIS VARIABLE IS
*					   			   ALLOCATED AND INNITIALIZED WITH ZERO
*								   BEFORE CALLING THIS FUNCTION WITH A SIZE
*								   OF "p" TIMES "4" 'DOUBLE' MEMORY SPACES.
*								   Note that the results will be stored in
*								   ascending order with respect to the
*								   outputs of the system under study. In
*								   other words, from the first output in row
*								   index "0" up to the last output in row
*								   index "p-1". Moreover, each individual
*								   output result will contain four columns
*								   where the "true positives" will be stored
*								   in the column index 0; the
*								   "false positives" will be stored in the
*								   column index 1; the "false negatives"
*								   will be stored in the column index 2 and;
*								   the "true negatives" will be stored in
*								   the column index 3.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "confusionMatrix".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: N/A
*/
void getConfusionMatrix(double *realOutputMatrix, double *predictedOutputMatrix, int n, int p, double *confusionMatrix) {
	// We calculate the confusion matrix between the argument pointer variables "realOutputMatrix" and "predictedOutputMatrix".
	int currentRowTimesP; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int currentOutputTimesPtimesFour; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesP = currentRow*p;
		for (int currentOutput=0; currentOutput<p; currentOutput++) {
			currentRowAndColumn = currentOutput + currentRowTimesP;
			currentOutputTimesPtimesFour = currentOutput*p*4;
			if ((realOutputMatrix[currentRowAndColumn]==1) && (predictedOutputMatrix[currentRowAndColumn]==1)) {
				confusionMatrix[0 + currentOutputTimesPtimesFour] += 1; // Increase the true positive counter.
			} else if ((realOutputMatrix[currentRowAndColumn]==0) && (predictedOutputMatrix[currentRowAndColumn]==1)) {
				confusionMatrix[1 + currentOutputTimesPtimesFour] += 1; // Increase the false positive counter.
			} else if ((realOutputMatrix[currentRowAndColumn]==1) && (predictedOutputMatrix[currentRowAndColumn]==0)) {
				confusionMatrix[2 + currentOutputTimesPtimesFour] += 1; // Increase the false negative counter.
			} else if ((realOutputMatrix[currentRowAndColumn]==0) && (predictedOutputMatrix[currentRowAndColumn]==0)) {
				confusionMatrix[3 + currentOutputTimesPtimesFour] += 1; // Increase the true negative counter.
			} else {
				printf("\nERROR: An output value from either the real or the predicted output matrixes contains a non binary value in the row index %d. and output/column index %d\n", currentRow, currentOutput);
			}	
		}
	}
}


/**
* The "getAccuracy()" function is used to calculate and obtain the
* classification evaluation metric known as the accuracy. Such method
* will be applied with respect to the argument pointer variables
* "realOutputMatrix" and "predictedOutputMatrix". Then, its result
* will be stored in the argument pointer variable "accuracy".
* 
* @param double *realOutputMatrix - This argument will contain the
*							   		pointer to a memory allocated
*								    output matrix, representing
*									the real data of the system
*									under study. This variable will
*									be used as a reference to
*									calculate and obtain the
*									accuracy with respect to the
*									argument pointer variable
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
*										 used to calculate and obtain
*										 the accuracy. THIS VARIABLE
*										 SHOULD BE ALLOCATED AND
*										 INNITIALIZED BEFORE CALLING
*										 THIS FUNCTION WITH A SIZE OF
*										 "n" TIMES "p" 'DOUBLE'
*										 MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix, containing
*				 the real/predicted results of the system under study.
*
* @param double *accuracy - This argument will contain the pointer to a
*							memory allocated variable in which we will
*							store the resulting metric evaluation
*							obtained after having applied the accuracy
*							metric between the argument pointer variables
*							"realOutputMatrix" and
*							"predictedOutputMatrix". IT IS INDISPENSABLE
*							THAT THIS VARIABLE IS ALLOCATED AND
*							INNITIALIZED WITH ZERO BEFORE CALLING THIS
*							FUNCTION WITH A SIZE OF "p" TIMES "1"
*							'DOUBLE' MEMORY SPACES. Note that the results
*							will be stored in ascending order with
*							respect to the outputs of the system under
*							study. In other words, from the first output
*							in row index "0" up to the last output in row
*							index "p-1".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "accuracy".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: N/A
*/
void getAccuracy(double *realOutputMatrix, double *predictedOutputMatrix, int n, int p, double *accuracy) {
	// In order to calculate the accuracy, we calculate the true positives and true negatives between the argument pointer variables "realOutputMatrix" and "predictedOutputMatrix".
	int currentRowTimesP; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int currentOutputTimesPtimesTwo; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	double *tp_and_tn = (double *) calloc(p*2, sizeof(double)); // Variable used to store the true positives in column index 0 and true negatives in column index 1 for each of the outputs availabe, each one of them stored in ascending order from row index 0 up to row index p-1.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesP = currentRow*p;
		for (int currentOutput=0; currentOutput<p; currentOutput++) {
			currentRowAndColumn = currentOutput + currentRowTimesP;
			currentOutputTimesPtimesTwo = currentOutput*p*2;
			if ((realOutputMatrix[currentRowAndColumn]==1) && (predictedOutputMatrix[currentRowAndColumn]==1)) {
				tp_and_tn[0 + currentOutputTimesPtimesTwo] += 1; // Increase the true positive counter.
			} else if ((realOutputMatrix[currentRowAndColumn]==0) && (predictedOutputMatrix[currentRowAndColumn]==0)) {
				tp_and_tn[1 + currentOutputTimesPtimesTwo] += 1; // Increase the true negative counter.
			}
		}
	}
	
	// We calculate the accuracy for every output given.
	for (int currentOutput=0; currentOutput<p; currentOutput++) {
		currentOutputTimesPtimesTwo = currentOutput*p*2;
		accuracy[currentOutput] = (tp_and_tn[0 + currentOutputTimesPtimesTwo] + tp_and_tn[1 + currentOutputTimesPtimesTwo]) / n;
	}
	
	// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
	free(tp_and_tn);
}


/**
* The "getPrecision()" function is used to calculate and obtain the
* classification evaluation metric known as the precision. Such method
* will be applied with respect to the argument pointer variables
* "realOutputMatrix" and "predictedOutputMatrix". Then, its result
* will be stored in the argument pointer variable "precision".
* 
* @param double *realOutputMatrix - This argument will contain the
*							   		pointer to a memory allocated
*								    output matrix, representing
*									the real data of the system
*									under study. This variable will
*									be used as a reference to
*									calculate and obtain the
*									precision with respect to the
*									argument pointer variable
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
*										 used to calculate and obtain
*										 the precision. THIS VARIABLE
*										 SHOULD BE ALLOCATED AND
*										 INNITIALIZED BEFORE CALLING
*										 THIS FUNCTION WITH A SIZE OF
*										 "n" TIMES "p" 'DOUBLE'
*										 MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix, containing
*				 the real/predicted results of the system under study.
*
* @param double *precision - This argument will contain the pointer to a
*							memory allocated variable in which we will
*							store the resulting metric evaluation
*							obtained after having applied the precision
*							metric between the argument pointer variables
*							"realOutputMatrix" and
*							"predictedOutputMatrix". IT IS INDISPENSABLE
*							THAT THIS VARIABLE IS ALLOCATED AND
*							INNITIALIZED WITH ZERO BEFORE CALLING THIS
*							FUNCTION WITH A SIZE OF "p" TIMES "1"
*							'DOUBLE' MEMORY SPACES. Note that the results
*							will be stored in ascending order with
*							respect to the outputs of the system under
*							study. In other words, from the first output
*							in row index "0" up to the last output in row
*							index "p-1".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "precision".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: NOVEMBER 26, 2021
*/
void getPrecision(double *realOutputMatrix, double *predictedOutputMatrix, int n, int p, double *precision) {
	// In order to calculate the precision, we temporarly store the sum of the product of "realOutputMatrix" and "predictedOutputMatrix" for each available sample in the variable "precision". In addition, we also calculate the sum of the "predictedOutputMatrix".
	int currentRowTimesP; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	double *sumOfPredictedOutputMatrix = (double *) calloc(p, sizeof(double)); // Variable used to store the total sum of values contained in the "predictedOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesP = currentRow*p;
		for (int currentOutput=0; currentOutput<p; currentOutput++) {
			currentRowAndColumn = currentOutput + currentRowTimesP;
			precision[currentOutput] += (realOutputMatrix[currentRowAndColumn] * predictedOutputMatrix[currentRowAndColumn]);
			sumOfPredictedOutputMatrix[currentOutput] += predictedOutputMatrix[currentRowAndColumn];
		}
	}
	
	// We calculate the precision for every output given.
	for (int currentOutput=0; currentOutput<p; currentOutput++) {
		if (sumOfPredictedOutputMatrix[currentOutput] == 0) {
			printf("ERROR: The total amount of true positives and false positives from the output column index %d equals zero. Therefore, the precision cannot be calculated because it gives a mathematical indetermination.\n", currentOutput);
		}
		precision[currentOutput] = precision[currentOutput] / sumOfPredictedOutputMatrix[currentOutput];
	}
	
	// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
	free(sumOfPredictedOutputMatrix);
}


/**
* The "getRecall()" function is used to calculate and obtain the
* classification evaluation metric known as the precision. Such method
* will be applied with respect to the argument pointer variables
* "realOutputMatrix" and "predictedOutputMatrix". Then, its result
* will be stored in the argument pointer variable "precision".
* 
* @param double *realOutputMatrix - This argument will contain the
*							   		pointer to a memory allocated
*								    output matrix, representing
*									the real data of the system
*									under study. This variable will
*									be used as a reference to
*									calculate and obtain the
*									precision with respect to the
*									argument pointer variable
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
*										 used to calculate and obtain
*										 the precision. THIS VARIABLE
*										 SHOULD BE ALLOCATED AND
*										 INNITIALIZED BEFORE CALLING
*										 THIS FUNCTION WITH A SIZE OF
*										 "n" TIMES "p" 'DOUBLE'
*										 MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix, containing
*				 the real/predicted results of the system under study.
*
* @param double *precision - This argument will contain the pointer to a
*							memory allocated variable in which we will
*							store the resulting metric evaluation
*							obtained after having applied the precision
*							metric between the argument pointer variables
*							"realOutputMatrix" and
*							"predictedOutputMatrix". IT IS INDISPENSABLE
*							THAT THIS VARIABLE IS ALLOCATED AND
*							INNITIALIZED WITH ZERO BEFORE CALLING THIS
*							FUNCTION WITH A SIZE OF "p" TIMES "1"
*							'DOUBLE' MEMORY SPACES. Note that the results
*							will be stored in ascending order with
*							respect to the outputs of the system under
*							study. In other words, from the first output
*							in row index "0" up to the last output in row
*							index "p-1".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "precision".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: NOVEMBER 26, 2021
*/
void getRecall(double *realOutputMatrix, double *predictedOutputMatrix, int n, int p, double *recall) {
	// In order to calculate the recall, we temporarly store the sum of the product of "realOutputMatrix" and "predictedOutputMatrix" for each available sample in the variable "recall". In addition, we also calculate the sum of the "realOutputMatrix".
	int currentRowTimesP; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	double *sumOfRealOutputMatrix = (double *) calloc(p, sizeof(double)); // Variable used to store the total sum of values contained in the "realOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesP = currentRow*p;
		for (int currentOutput=0; currentOutput<p; currentOutput++) {
			currentRowAndColumn = currentOutput + currentRowTimesP;
			recall[currentOutput] += (realOutputMatrix[currentRowAndColumn] * predictedOutputMatrix[currentRowAndColumn]);
			sumOfRealOutputMatrix[currentOutput] += realOutputMatrix[currentRowAndColumn];
		}
	}
	
	// We calculate the recall for every output given.
	for (int currentOutput=0; currentOutput<p; currentOutput++) {
		if (sumOfRealOutputMatrix[currentOutput] == 0) {
			printf("ERROR: The total amount of true positives and false negatives from the output column index %d equals zero. Therefore, the recall cannot be calculated because it gives a mathematical indetermination.\n", currentOutput);
		}
		recall[currentOutput] = recall[currentOutput] / sumOfRealOutputMatrix[currentOutput];
	}
	
	// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
	free(sumOfRealOutputMatrix);
}


/**
* The "getF1score()" function is used to calculate and obtain the
* classification evaluation metric known as the F1 score. Such method
* will be applied with respect to the argument pointer variables
* "realOutputMatrix" and "predictedOutputMatrix". Then, its result
* will be stored in the argument pointer variable "F1 score".
* 
* @param double *realOutputMatrix - This argument will contain the
*							   		pointer to a memory allocated
*								    output matrix, representing
*									the real data of the system
*									under study. This variable will
*									be used as a reference to
*									calculate and obtain the
*									F1 score with respect to the
*									argument pointer variable
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
*										 used to calculate and obtain
*										 the F1 score. THIS VARIABLE
*										 SHOULD BE ALLOCATED AND
*										 INNITIALIZED BEFORE CALLING
*										 THIS FUNCTION WITH A SIZE OF
*										 "n" TIMES "p" 'DOUBLE'
*										 MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix, containing
*				 the real/predicted results of the system under study.
*
* @param double *F1score - This argument will contain the pointer to a
*						   memory allocated variable in which we will
*						   store the resulting metric evaluation
*						   obtained after having applied the F1 score
*						   metric between the argument pointer variables
*						   "realOutputMatrix" and "predictedOutputMatrix".
*						   IT IS INDISPENSABLE THAT THIS VARIABLE IS
*						   ALLOCATED AND INNITIALIZED WITH ZERO BEFORE
*						   CALLING THIS FUNCTION WITH A SIZE OF "p" TIMES
*						   "1" 'DOUBLE' MEMORY SPACES. Note that the
*						   results will be stored in ascending order with
*						   respect to the outputs of the system under
*						   study. In other words, from the first output
*						   in row index "0" up to the last output in row
*						   index "p-1".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "F1score".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: N/A
*/
void getF1score(double *realOutputMatrix, double *predictedOutputMatrix, int n, int p, double *F1score) {
	// In order to calculate the F1score, we first calculate what is described in the following notes:
	// NOTE: We will temporarly store the sum of the product of "realOutputMatrix" and "predictedOutputMatrix" for each available sample in the variable "precision".
	// NOTE: we will calculate the sum of the "predictedOutputMatrix" and temporarly store it in the variable "F1score", for performance purposes.
	// NOTE: we will calculate the sum of the "realOutputMatrix" and temporarly store it in the variable "recall", for performance purposes.
	int currentRowTimesP; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	double *precision = (double *) calloc(p, sizeof(double)); // Variable used to store the precision with respect to the output data given in "realOutputMatrix" and "predictedOutputMatrix".
	double *recall = (double *) calloc(p, sizeof(double)); // Variable used to store the recall with respect to the output data given in "realOutputMatrix" and "predictedOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesP = currentRow*p;
		for (int currentOutput=0; currentOutput<p; currentOutput++) {
			currentRowAndColumn = currentOutput + currentRowTimesP;
			precision[currentOutput] += (realOutputMatrix[currentRowAndColumn] * predictedOutputMatrix[currentRowAndColumn]);
			F1score[currentOutput] += predictedOutputMatrix[currentRowAndColumn];
			recall[currentOutput] += realOutputMatrix[currentRowAndColumn];
		}
	}
	
	// We conclude the calculation of the precision and recall for every output given.
	for (int currentOutput=0; currentOutput<p; currentOutput++) {
		if (recall[currentOutput] == 0) {
			printf("ERROR: The total amount of true positives and false negatives from the output column index %d equals zero. Therefore, the F1 score cannot be calculated because the recall gives a mathematical indetermination.\n", currentOutput);
		}
		recall[currentOutput] = precision[currentOutput] / recall[currentOutput];
		if (F1score[currentOutput] == 0) {
			printf("ERROR: The total amount of true positives and false positives from the output column index %d equals zero. Therefore, the F1 score cannot be calculated because the precision gives a mathematical indetermination.\n", currentOutput);
		}
		precision[currentOutput] = precision[currentOutput] / F1score[currentOutput];
	}
	
	// We calculate the F1 score for every output given
	for (int currentOutput=0; currentOutput<p; currentOutput++) {
		F1score[currentOutput] = 2*precision[currentOutput]*recall[currentOutput] / (precision[currentOutput] + recall[currentOutput]);
	}
	
	// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
	free(precision);
	free(recall);
}
