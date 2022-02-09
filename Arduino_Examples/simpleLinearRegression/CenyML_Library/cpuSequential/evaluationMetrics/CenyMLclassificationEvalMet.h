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
#ifndef CENYMLCLASSIFICATIONEVALMET_H
#define CENYMLCLASSIFICATIONEVALMET_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void getCrossEntropyError(float *, float *, int, float, float *);
void getConfusionMatrix(float *, float *, int, float *);
void getAccuracy(float *, float *, int, float *);
void getPrecision(float *, float *, int, float *);
void getRecall(float *, float *, int, float *);
void getF1score(float *, float *, int, float *);




/**
* The "getCrossEntropyError()" function is used to apply a
* classification evaluation metric known as the cross entropy error.
* Such method will be applied with respect to the argument pointer
* variables "realOutputMatrix" and "predictedOutputMatrix". Then, its
* result will be stored in the argument pointer variable "NLL".
* 
* @param float *realOutputMatrix - This argument will contain the
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
*									TIMES "p" 'float' MEMORY SPACES.
*
* @param float *predictedOutputMatrix - This argument will contain
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
*										 'float' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param float NLLepsilon - This argument will contain the user defined
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
* @param float *NLL - This argument will contain the pointer to a
*					   memory allocated variable in which we will store
*					   the resulting metric evaluation obtained after
*					   having applied the cross entropy error metric
*					   between the argument pointer variables
*					   "realOutputMatrix" and "predictedOutputMatrix".
*					   IT IS INDISPENSABLE THAT THIS VARIABLE IS
*					   ALLOCATED AND INNITIALIZED WITH ZERO BEFORE
*					   CALLING THIS FUNCTION WITH A SIZE OF "1" 'float'
*					   MEMORY SPACES, IN WHICH THE RESULT WILL BE STORED.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "NLL".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 22, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getCrossEntropyError(float *realOutputMatrix, float *predictedOutputMatrix, int n, float NLLepsilon, float *NLL) {
	// We calculate the cross entropy error between the argument pointer variables "realOutputMatrix" and "predictedOutputMatrix".
	float oneMinusEpsilon = 1 - NLLepsilon; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	float rOM; // This variable is used to store the current value of the real output matrix in which, if the value if zero, it will be replaced with the value of the argument variable "NLLepsilon".
	float pOM; // This variable is used to store the current value of the predicted output matrix in which, if the value if zero, it will be replaced with the value of the argument variable "NLLepsilon".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		// We temporarly store the sum the user defined epsilon value in "NLLepsilon" to any "0" value and substract it to any "1" value of the real output matrix.
		if (realOutputMatrix[currentRow] == 0) {
			rOM = NLLepsilon;
		} else if (realOutputMatrix[currentRow] == 1) {
			rOM = oneMinusEpsilon;
		} else {
			rOM = realOutputMatrix[currentRow];
		}
		
		// We temporarly store the sum the user defined epsilon value in "NLLepsilon" to any "0" value and substract it to any "1" value of the predicted output matrix.
		if (predictedOutputMatrix[currentRow] == 0) {
			pOM = NLLepsilon;
		} else if (predictedOutputMatrix[currentRow] == 1) {
			pOM = oneMinusEpsilon;
		} else {
			pOM = predictedOutputMatrix[currentRow];
		}
		
		// We calculate the current error and add it into the memory space were it will be stored.
		NLL[0] = NLL[0] - rOM*log(pOM) - (1-rOM) * log(1-pOM);
	}
	
	return;
}


/**
* The "getConfusionMatrix()" function is used to calculate and obtain
* the classification evaluation metric known as the confusion matrix.
* Such method will be applied with respect to the argument pointer
* variables "realOutputMatrix" and "predictedOutputMatrix". Then, its
* result will be stored in the argument pointer variable
* "confusionMatrix".
* 
* @param float *realOutputMatrix - This argument will contain the
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
*									TIMES "p" 'float' MEMORY SPACES.
*
* @param float *predictedOutputMatrix - This argument will contain
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
*										 'float' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param float *confusionMatrix - This argument will contain the pointer
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
*								   OF "4" TIMES "1" 'float' MEMORY SPACES,
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
* LAST UPDATE: DECEMBER 04, 2021
*/
void getConfusionMatrix(float *realOutputMatrix, float *predictedOutputMatrix, int n, float *confusionMatrix) {
	// We calculate the confusion matrix between the argument pointer variables "realOutputMatrix" and "predictedOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		if ((realOutputMatrix[currentRow]==1) && (predictedOutputMatrix[currentRow]==1)) {
			confusionMatrix[0] += 1; // Increase the true positive counter.
		} else if ((realOutputMatrix[currentRow]==0) && (predictedOutputMatrix[currentRow]==1)) {
			confusionMatrix[1] += 1; // Increase the false positive counter.
		} else if ((realOutputMatrix[currentRow]==1) && (predictedOutputMatrix[currentRow]==0)) {
			confusionMatrix[2] += 1; // Increase the false negative counter.
		} else if ((realOutputMatrix[currentRow]==0) && (predictedOutputMatrix[currentRow]==0)) {
			confusionMatrix[3] += 1; // Increase the true negative counter.
		} else {
			Serial.print("\nERROR: An output value from either the real or the predicted output matrixes contains a non binary value.\n");
		}
	}
	
	return;
}


/**
* The "getAccuracy()" function is used to calculate and obtain the
* classification evaluation metric known as the accuracy. Such method
* will be applied with respect to the argument pointer variables
* "realOutputMatrix" and "predictedOutputMatrix". Then, its result
* will be stored in the argument pointer variable "accuracy".
* 
* @param float *realOutputMatrix - This argument will contain the
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
*									TIMES "p" 'float' MEMORY SPACES.
*
* @param float *predictedOutputMatrix - This argument will contain
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
*										 "n" TIMES "p" 'float'
*										 MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param float *accuracy - This argument will contain the pointer to a
*							memory allocated variable in which we will
*							store the resulting metric evaluation
*							obtained after having applied the accuracy
*							metric between the argument pointer variables
*							"realOutputMatrix" and
*							"predictedOutputMatrix". IT IS INDISPENSABLE
*							THAT THIS VARIABLE IS ALLOCATED AND
*							INNITIALIZED WITH ZERO BEFORE CALLING THIS
*							FUNCTION WITH A SIZE OF "1" 'float' MEMORY
*							SPACES where the result will be stored.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "accuracy".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getAccuracy(float *realOutputMatrix, float *predictedOutputMatrix, int n, float *accuracy) {
	// In order to calculate the accuracy, we calculate the true positives and true negatives between the argument pointer variables "realOutputMatrix" and "predictedOutputMatrix".
	float tp = 0; // Variable used to store the true positives.
	float tn = 0; // Variable used to store the true negatives.
	for (int currentRow = 0; currentRow < n; currentRow++) {
		if ((realOutputMatrix[currentRow]==1) && (predictedOutputMatrix[currentRow]==1)) {
			tp += 1; // Increase the true positive counter.
		} else if ((realOutputMatrix[currentRow]==0) && (predictedOutputMatrix[currentRow]==0)) {
			tn += 1; // Increase the true negative counter.
		}
	}
	accuracy[0] = (tp + tn) / n; // We apply the last procedure to mathematically obtain the accuracy.
	
	return;
}


/**
* The "getPrecision()" function is used to calculate and obtain the
* classification evaluation metric known as the precision. Such method
* will be applied with respect to the argument pointer variables
* "realOutputMatrix" and "predictedOutputMatrix". Then, its result
* will be stored in the argument pointer variable "precision".
* 
* @param float *realOutputMatrix - This argument will contain the
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
*									TIMES "p" 'float' MEMORY SPACES.
*
* @param float *predictedOutputMatrix - This argument will contain
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
*										 "n" TIMES "p" 'float'
*										 MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param float *precision - This argument will contain the pointer to a
*							memory allocated variable in which we will
*							store the resulting metric evaluation
*							obtained after having applied the precision
*							metric between the argument pointer variables
*							"realOutputMatrix" and
*							"predictedOutputMatrix". IT IS INDISPENSABLE
*							THAT THIS VARIABLE IS ALLOCATED AND
*							INNITIALIZED WITH ZERO BEFORE CALLING THIS
*							FUNCTION WITH A SIZE OF "1" 'float' MEMORY
*							SPACES where the result will be stored.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "precision".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getPrecision(float *realOutputMatrix, float *predictedOutputMatrix, int n, float *precision) {
	// In order to calculate the precision, we temporarly store the sum of the product of "realOutputMatrix" and "predictedOutputMatrix" for each available sample in the variable "precision". In addition, we also calculate the sum of the "predictedOutputMatrix".
	float sumOfPredictedOutputMatrix = 0; // Variable used to store the total sum of values contained in the "predictedOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		precision[0] += (realOutputMatrix[currentRow] * predictedOutputMatrix[currentRow]);
		sumOfPredictedOutputMatrix += predictedOutputMatrix[currentRow];
	}
	
	// We calculate the precision for every output given.
	if (sumOfPredictedOutputMatrix == 0) {
		Serial.print("ERROR: The total amount of true positives and false positives equals zero. Therefore, the precision cannot be calculated because it gives a mathematical indetermination.\n");
	}
	precision[0] = precision[0] / sumOfPredictedOutputMatrix;
	
	return;
}


/**
* The "getRecall()" function is used to calculate and obtain the
* classification evaluation metric known as the precision. Such method
* will be applied with respect to the argument pointer variables
* "realOutputMatrix" and "predictedOutputMatrix". Then, its result
* will be stored in the argument pointer variable "precision".
* 
* @param float *realOutputMatrix - This argument will contain the
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
*									TIMES "p" 'float' MEMORY SPACES.
*
* @param float *predictedOutputMatrix - This argument will contain
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
*										 "n" TIMES "p" 'float'
*										 MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param float *precision - This argument will contain the pointer to a
*							memory allocated variable in which we will
*							store the resulting metric evaluation
*							obtained after having applied the precision
*							metric between the argument pointer variables
*							"realOutputMatrix" and
*							"predictedOutputMatrix". IT IS INDISPENSABLE
*							THAT THIS VARIABLE IS ALLOCATED AND
*							INNITIALIZED WITH ZERO BEFORE CALLING THIS
*							FUNCTION WITH A SIZE OF "1" 'float' MEMORY
*							SPACES where the result will be stored.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "precision".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getRecall(float *realOutputMatrix, float *predictedOutputMatrix, int n, float *recall) {
	// In order to calculate the recall, we temporarly store the sum of the product of "realOutputMatrix" and "predictedOutputMatrix" for each available sample in the variable "recall". In addition, we also calculate the sum of the "realOutputMatrix".
	float sumOfRealOutputMatrix = 0; // Variable used to store the total sum of values contained in the "realOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		recall[0] += (realOutputMatrix[currentRow] * predictedOutputMatrix[currentRow]);
		sumOfRealOutputMatrix += realOutputMatrix[currentRow];
	}
	
	// We calculate the recall for every output given.
	if (sumOfRealOutputMatrix == 0) {
		Serial.print("ERROR: The total amount of true positives and false negatives equals zero. Therefore, the recall cannot be calculated because it gives a mathematical indetermination.\n");
	}
	recall[0] = recall[0] / sumOfRealOutputMatrix;
}


/**
* The "getF1score()" function is used to calculate and obtain the
* classification evaluation metric known as the F1 score. Such method
* will be applied with respect to the argument pointer variables
* "realOutputMatrix" and "predictedOutputMatrix". Then, its result
* will be stored in the argument pointer variable "F1 score".
* 
* @param float *realOutputMatrix - This argument will contain the
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
*									TIMES "p" 'float' MEMORY SPACES.
*
* @param float *predictedOutputMatrix - This argument will contain
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
*										 "n" TIMES "p" 'float'
*										 MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param float *F1score - This argument will contain the pointer to a
*						   memory allocated variable in which we will
*						   store the resulting metric evaluation
*						   obtained after having applied the F1 score
*						   metric between the argument pointer variables
*						   "realOutputMatrix" and "predictedOutputMatrix".
*						   IT IS INDISPENSABLE THAT THIS VARIABLE IS
*						   ALLOCATED AND INNITIALIZED WITH ZERO BEFORE
*						   CALLING THIS FUNCTION WITH A SIZE OF "1"
*						   'float' MEMORY SPACES where the result will
*						   be stored.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "F1score".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 23, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
void getF1score(float *realOutputMatrix, float *predictedOutputMatrix, int n, float *F1score) {
	// In order to calculate the F1score, we first calculate what is described in the following notes:
	// NOTE: We will temporarly store the sum of the product of "realOutputMatrix" and "predictedOutputMatrix" for each available sample in the variable "precision".
	// NOTE: we will calculate the sum of the "predictedOutputMatrix" and temporarly store it in the variable "F1score", for performance purposes.
	// NOTE: we will calculate the sum of the "realOutputMatrix" and temporarly store it in the variable "recall", for performance purposes.
	float precision = 0; // Variable used to store the precision with respect to the output data given in "realOutputMatrix" and "predictedOutputMatrix".
	float recall = 0; // Variable used to store the recall with respect to the output data given in "realOutputMatrix" and "predictedOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		precision += (realOutputMatrix[currentRow] * predictedOutputMatrix[currentRow]);
		F1score[0] += predictedOutputMatrix[currentRow];
		recall += realOutputMatrix[currentRow];
	}
	
	// We conclude the calculation of the precision and recall for every output given.
	if (recall == 0) {
		Serial.print("ERROR: The total amount of true positives and false negatives equals zero. Therefore, the F1 score cannot be calculated because the recall gives a mathematical indetermination.\n");
	}
	recall = precision / recall;
	if (F1score[0] == 0) {
		Serial.print("ERROR: The total amount of true positives and false positives equals zero. Therefore, the F1 score cannot be calculated because the precision gives a mathematical indetermination.\n");
	}
	precision = precision / F1score[0];
	
	// We calculate the F1 score for every output given
	F1score[0] = 2*precision*recall / (precision + recall);
	
	return;
}



#endif

