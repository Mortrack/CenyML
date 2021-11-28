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
#include "CenyMLdeepLearning.h"
// IMPORTANT NOTE: This library uses the math.h library and therefore,
//				   remember to use the "-lm" flag when compiling it.


/**
* The "getSingleNeuronDNN()" function is used to apply the
* machine learning algorithm called simple linear regression. Within
* this process, the best fitting equation with the form of "y_hat =
* b_0 + b_1*x" will be identified with respect to the sampled data
* given through the argument pointer variables "X" and "Y". As a
* result, the identified coefficient values will be stored in the
* argument pointer variable "b".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INNITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m=1" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INNITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A SIZE OF
*					 "n" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable in which we will store
*					 the identified best fitting coefficient values
*					 for the desired machine learning algorithm. These
*					 coefficients will each be stored in the same
*					 column but under different rows where the first
*					 coefficient (b_0) will be stored in the row with
*					 index 0 and the last coefficient (b_1) will be
*					 stored in the row with index 1. IT IS
*					 INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*					 BEFORE CALLING THIS FUNCTION WITH A VARIABLE SIZE
*					 OF "m+1=2" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 28, 2021
* LAST UPDATE: N/A
*/
// TODO: It is still pending to add the functionality of neuron->isReportLearningProgress.
void getSingleNeuronDNN(struct singleNeuronDnnStruct *neuron) {
	// If the machine learning samples are less than value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->n < 1) {
		printf("\nERROR: The machine learning samples must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->m < 1) {
		printf("\nERROR: The machine learning features (independent variables) must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study exceed the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->p != 1) {
		printf("\nERROR: The outputs of the system under study must be equal to 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the identifier assigned to "neuron->activationFunctionToBeUsed" is not in the range of 0 and 11, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->activationFunctionToBeUsed>11) && (neuron->activationFunctionToBeUsed<0)) {
		printf("\nERROR: The defined activation function identifier assigned to \"activationFunctionToBeUsed\" in the struct of \"singleNeuronDnnStruct\" must be a whole value in the range of 0 to 11. Please add a valid identifier number to it.\n");
		exit(1);
	}
	// If the flag "neuron->isClassification" has a value different of 0 and 1, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->isClassification!=0) && (neuron->isClassification!=1)) {
		printf("\nERROR: The defined value for the flag \"isClassification\" in the struct of \"singleNeuronDnnStruct\" can only have a value of either 0 or 1. Please add a valid value to it.\n");
		exit(1);
	}
	// If the flag "neuron->isInitial_w" has a value different of 0 and 1, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->isInitial_w!=1) && (neuron->isInitial_w!=0)) {
		printf("\nERROR: The defined value for the flag \"isInitial_w\" in the struct of \"singleNeuronDnnStruct\" can only have a value of either 0 or 1. Please add a valid value to it.\n");
		exit(1);
	}
	// If the flag "neuron->isReportLearningProgress" has a value different of 0 and 1, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->isReportLearningProgress!=1) && (neuron->isReportLearningProgress!=0)) {
		printf("\nERROR: The defined value for the flag \"isReportLearningProgress\" in the struct of \"singleNeuronDnnStruct\" can only have a value of either 0 or 1. Please add a valid value to it.\n");
		exit(1);
	}
	// If the requested epochs are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->maxEpochs < 1) {
		printf("\nERROR: The defined value for \"maxEpochs\" in the struct of \"singleNeuronDnnStruct\" must be equal or greater than 1 for this particular algorithm. Please add a valid value to it.\n");
		exit(1);
	}
	// If the machine learning features exceed the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->reportEachSpecifiedEpochs<1) && (neuron->maxEpochs<neuron->reportEachSpecifiedEpochs)) {
		printf("\nERROR: The defined value for \"reportEachSpecifiedEpochs\" in the struct of \"singleNeuronDnnStruct\" cannot be less than 1 and cannot be greater than the value of \"maxEpochs\" in the struct of \"singleNeuronDnnStruct\". Please add a valid value to it.\n");
		exit(1);
	}
	// If the value of "neuron->stopAboveThisAccuracy" is not in the range of 0 and 1, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->stopAboveThisAccuracy<0) && (neuron->stopAboveThisAccuracy>1)) {
		printf("\nERROR: The defined value for the flag \"stopAboveThisAccuracy\" in the struct of \"singleNeuronDnnStruct\" can only have a value in the range of 0 and 1. Please add a valid value to it.\n");
		exit(1);
	}
	
	// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
	// Store the data that must be contained in the input matrix "X_tilde". In addition, we obtain the transpose of "X_tilde".
	int currentRowTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
	int mPlusOne = m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	double *X_tilde = (double *) malloc(n*mPlusOne*sizeof(double)); // This variable will contain the input data of the system under study ("X") and an additional first row with values of "1".
	double *TransposeOf_X_tilde = (double *) malloc(mPlusOne*n*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
	for (int currentRow=0; currentRow<n; currentRow++) {
		currentRow2 = 0; // We reset the counters used in the following for-loop.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowTimesM = currentRow*m;
		X_tilde[currentRowTimesMplusOne] = 1;
		TransposeOf_X_tilde[currentColumn2] = X_tilde[currentRowTimesMplusOne];
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			currentRowAndColumn = currentColumn + currentRowTimesMplusOne;
			X_tilde[currentRowAndColumn] = X[currentColumn-1 + currentRowTimesM];
			currentRow2++;
			TransposeOf_X_tilde[currentColumn2 + currentRow2*n] = X_tilde[currentRowAndColumn];
		}
		currentColumn2++;
	}
	
	// -------------------- WEIGHT INITIALIZATION -------------------- //
	// Store the initial weight values into "neuron->w_new".
	if (neuron->isInitial_w == 0) {
		// In order to initialize "neuron->w_new" with random values between "-1" and "+1"., intialize random number generator.
	    time_t t;
		srand((unsigned) time(&t));
	    
	    // Initialize "neuron->w_new" with random values between -1 to +1 with three decimals at the most.
	    double currentRandomNumber;
	    for (int current_w=0 ; current_w<(m+1); current_w++) {
	        currentRandomNumber = ((float) (rand() % 1000))/500 - 1;
	        neuron->w_first[current_w] = currentRandomNumber;
	        neuron->w_new[current_w] = currentRandomNumber;
	    }
	} else if (neuron->isInitial_w == 1) {
		// Pass the values of "neuron->w_first" to "neuron->w_new".
		for (int current_w=0 ; current_w<(m+1); current_w++) {
	        neuron->w_new[current_w] = neuron->w_first[current_w];
	    }
	} else {
		// If the value assigned to "neuron->isInitial_w" is different than 0 and 1, then emit an error message and terminate the program. Otherwise, continue with the program.
		printf("\nERROR: The valid values that can be assigned to \"neuron->isInitial_w\" are either the binary value 1 or 0. Please assign one of those to it.\n");
		exit(1);
	}
	
	// Determine if the requested model to generate is meant for a classification or for a regression problem to then solve it accordingly.
	if (neuron->isClassification == 1) {
		// ----------------------------------------- //
		// ----- CLASSIFICATION MODEL SELECTED ----- //
		// ----------------------------------------- //


		// ----------- SOLUTION OF THE FIRST EPOCH OF THE MODEL ---------- //
		// We calculate the currently predicted output data made by the body of the neuron and store it in "f_x_tilde".
		double *f_x_tilde = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "f_x_tilde", which will contain the currently predicted output data made by the body of the neuron.
		for (int currentRow=0; currentRow<neuron->n; currentRow++) {
			f_x_tilde[currentRow] = neuron->w_new[0];
			currentRowTimesM = currentRow*m;
			for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
				f_x_tilde[currentRow] = f_x_tilde[currentRow] + neuron->w_new[currentColumn]*X_tilde[currentColumn + currentRowTimesM];
			}
		}
		
		// We calculate, in its continous (regression) form, the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
		// NOTE: "activationFunctions" is a pointer to each of the individual activation functions that were developed as static void functions.
	    static void (*activationFunctions[])(double *, double *, struct singleNeuronDnnStruct *) = {getReluActivation, getTanhActivation, getLogisticActivation, getRaiseToTheFirstPowerActivation, getSquareRootActivation, getRaiseToTheSecondPowerActivation, getRaiseToTheThirdPowerActivation, getRaiseToTheFourthPowerActivation, getRaiseToTheFifthPowerActivation, getRaiseToTheSixthPowerActivation, getFirstOrderDegreeExponentialActivation, getSecondOrderDegreeExponentialActivation};
	    double *A_u = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "A_u", which will contain the currently predicted output data made by the neuron.
		(*activationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, neuron); // We calculate A(u) and store it in the pointer variable "A_u".
		
		// We calculate the derivative of A(u).
		// NOTE: "derivateOfActivationFunctions" is a pointer to each of the individual derivatives of the activation functions that were developed as static void functions.
		static void (*derivateOfActivationFunctions[])(double *, double *, double *, struct singleNeuronDnnStruct *) = {getDerivateReluActivation, getDerivateTanhActivation, getLogisticActivation, getDerivateRaiseToTheFirstPowerActivation, getDerivateSquareRootActivation, getDerivateRaiseToTheSecondPowerActivation, getDerivateRaiseToTheThirdPowerActivation, getDerivateRaiseToTheFourthPowerActivation, getDerivateRaiseToTheFifthPowerActivation, getDerivateRaiseToTheSixthPowerActivation, getDerivateFirstOrderDegreeExponentialActivation, getDerivateSecondOrderDegreeExponentialActivation};
	    double *dA_u = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "dA_u", which will contain the derivative of A(u).
	    (*derivateOfActivationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, dA_u, neuron); // We calculate the derivative of A(u) and store it in the pointer variable "dA_u".
		// NOTE: Remember that "Y_hat" = A(u) = "A_u".
		
		// We apply the threshold define by the implementer in order to obtain a classification output and store it in "A_u".
		for (int currentRow=0; currentRow<neuron->n; currentRow++) {
			if (A_u[currentRow > neuron->threshold]) { // For performance purposes and compatibility with the accuracy method to be used, the classification output results will be either 1 or 0.
				A_u[currentRow] = 1;
			} else {
				A_u[currentRow] = 0;
			}
		}
		
		// We calculate the corresponding evaluation metric with respect to the actual data of the system under study "neuron->Y" and the currently predicted output made by the neuron "A_u".
		double *currentAccuracy = (double *) calloc(neuron->p*1, sizeof(double)); // Allocate the memory required for the variable "currentAccuracy", which will contain the current accuracy of the neuron.
		getAccuracy(neuron->Y, A_u, neuron->n, neuron->p, currentAccuracy); // We calculate the current accuracy of the neuron.
		
		// If the desired accuracy has been reached, then conclude the training process of the neuron. Otherwise, continue training it.
		if (currentAccuracy[0] > neuron->stopAboveThisAccuracy) {
			prinft("\nThe accuracy of the neuron has achieved a higher one with respect to the one that was specified as a goal the very first instant it was created.\n";
			return;
		}
		
		// -------- SOLUTION OF THE REMAINING EPOCHS OF THE MODEL -------- //
		double *w_old = (double *) malloc((neuron->m+1)*neuron->p*sizeof(double)); // Allocate the memory required for the variable "w_old", which will contain the previous weight values that were obtained with respect to the current ones.
		double *errorTerm = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "errorTerm", which will contain the current error term value to be taken into consideration for the update of the weight values.
		double *errorTerm_dot_Xtilde = (double *) malloc((neuron->m+1)*neuron->p*sizeof(double)); // Allocate the memory required for the variable "errorTerm_dot_Xtilde", which will contain the resulting dot product between the error term and the transpose of "X_tilde".
		int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		for (int currentEpoch=0; currentEpoch<(neuron->maxEpochs); currentEpoch++) {
			// Pass the data of "neuron->w_new" to "w_old".
			for (int currentCoefficient=0; currentCoefficient<(m+1); currentCoefficient++) {
				w_old[currentCoefficient] = neuron->w_new[currentCoefficient];
			}
			
			// Calculate the error term obtainable with the current weight values.
			for (int currentSample=0; currentSample<(neuron->n); currentSample++) {
				errorTerm[currentSample] = (neuron->Y[currentSample] - A_u[currentSample]) * dA_u[currentSample];
			}
			
			// We update the current weight values ("w_old") in order to obtain the new ones ("neuron->w_new").
			for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
				errorTerm_dot_Xtilde[currentRow] = 0;
				currentRowTimesN = currentRow*n;
				for (int currentSample=0; currentSample<(neuron->n); currentSample++) {
					// We first multiply all the samples of the "errorTerm" with all the samples of the transpose of "X_tilde".
					errorTerm_dot_Xtilde[currentRow] += errorTerm[currentSample] * TransposeOf_X_tilde[currentSample + currentRowTimesN];
				}
				// We now multiple the previous result with the learning rate and then update for the current weight value (which is indicated by "currentRow").
				neuron->w_new[currentRow] = w_old[currentRow] + neuron->learningRate * errorTerm_dot_Xtilde[currentRow];
			}
			
			// We recalculate the currently predicted output data made by the body of the neuron and store it in "f_x_tilde".
			for (int currentRow=0; currentRow<neuron->n; currentRow++) {
				f_x_tilde[currentRow] = neuron->w_new[0];
				currentRowTimesM = currentRow*m;
				for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
					f_x_tilde[currentRow] = f_x_tilde[currentRow] + neuron->w_new[currentColumn]*X_tilde[currentColumn + currentRowTimesM];
				}
			}
			
			// We recalculate the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
			(*activationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, neuron); // We calculate A(u) and store it in the pointer variable "A_u".
			
			// We recalculate the derivative of A(u).
		    (*derivateOfActivationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, dA_u, neuron); // We calculate the derivative of A(u) and store it in the pointer variable "dA_u".
			// NOTE: Remember that "Y_hat" = A(u) = "A_u".
			
			// We apply the threshold define by the implementer in order to obtain a classification output and store it in "A_u".
			for (int currentRow=0; currentRow<neuron->n; currentRow++) {
				if (A_u[currentRow > neuron->threshold]) { // For performance purposes and compatibility with the accuracy method to be used, the classification output results will be either 1 or 0.
					A_u[currentRow] = 1;
				} else {
					A_u[currentRow] = 0;
				}
			}
		
			// We recalculate the corresponding evaluation metric with respect to the actual data of the system under study "neuron->Y" and the currently predicted output made by the neuron "A_u".
			currentAccuracy[0] = 0; // We reset the value of this variable in order to recalculate it.
			getAccuracy(neuron->Y, A_u, neuron->n, neuron->p, currentAccuracy); // We calculate the current accuracy of the neuron.
			
			// If the desired accuracy has been reached, then conclude the training process of the neuron. Otherwise, continue training it.
			if (currentAccuracy[0] > neuron->stopAboveThisAccuracy) {
				prinft("\nThe accuracy of the neuron has achieved a higher one with respect to the one that was specified as a goal when concluding the epoch number %d.\n", currentEpoch);
				return;
			}
		}
	} else {
		// ------------------------------------- //
		// ----- REGRESSION MODEL SELECTED ----- //
		// ------------------------------------- //
		
		
		// ----------- SOLUTION OF THE FIRST EPOCH OF THE MODEL ---------- //
		// We calculate the currently predicted output data made by the body of the neuron and store it in "f_x_tilde".
		double *f_x_tilde = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "f_x_tilde", which will contain the currently predicted output data made by the body of the neuron.
		for (int currentRow=0; currentRow<neuron->n; currentRow++) {
			f_x_tilde[currentRow] = neuron->w_new[0];
			currentRowTimesM = currentRow*m;
			for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
				f_x_tilde[currentRow] = f_x_tilde[currentRow] + neuron->w_new[currentColumn]*X_tilde[currentColumn + currentRowTimesM];
			}
		}
		
		// We calculate the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
		// NOTE: "activationFunctions" is a pointer to each of the individual activation functions that were developed as static void functions.
	    static void (*activationFunctions[])(double *, double *, struct singleNeuronDnnStruct *) = {getReluActivation, getTanhActivation, getLogisticActivation, getRaiseToTheFirstPowerActivation, getSquareRootActivation, getRaiseToTheSecondPowerActivation, getRaiseToTheThirdPowerActivation, getRaiseToTheFourthPowerActivation, getRaiseToTheFifthPowerActivation, getRaiseToTheSixthPowerActivation, getFirstOrderDegreeExponentialActivation, getSecondOrderDegreeExponentialActivation};
	    double *A_u = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "A_u", which will contain the currently predicted output data made by the neuron.
		(*activationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, neuron); // We calculate A(u) and store it in the pointer variable "A_u".
		
		// We calculate the derivative of A(u).
		// NOTE: "derivateOfActivationFunctions" is a pointer to each of the individual derivatives of the activation functions that were developed as static void functions.
		static void (*derivateOfActivationFunctions[])(double *, double *, double *, struct singleNeuronDnnStruct *) = {getDerivateReluActivation, getDerivateTanhActivation, getLogisticActivation, getDerivateRaiseToTheFirstPowerActivation, getDerivateSquareRootActivation, getDerivateRaiseToTheSecondPowerActivation, getDerivateRaiseToTheThirdPowerActivation, getDerivateRaiseToTheFourthPowerActivation, getDerivateRaiseToTheFifthPowerActivation, getDerivateRaiseToTheSixthPowerActivation, getDerivateFirstOrderDegreeExponentialActivation, getDerivateSecondOrderDegreeExponentialActivation};
	    double *dA_u = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "dA_u", which will contain the derivative of A(u).
	    (*derivateOfActivationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, dA_u, neuron); // We calculate the derivative of A(u) and store it in the pointer variable "dA_u".
		// NOTE: Remember that "Y_hat" = A(u) = "A_u".
		
		// We calculate the corresponding evaluation metric with respect to the actual data of the system under study "neuron->Y" and the currently predicted output made by the neuron "A_u".
		double *currentAccuracy = (double *) calloc(neuron->p*1, sizeof(double)); // Allocate the memory required for the variable "currentAccuracy", which will contain the current accuracy of the neuron.
		getAdjustedCoefficientOfDetermination(neuron->Y, A_u, neuron->n, neuron->m, neuron->p, 1, currentAccuracy); // We calculate the current accuracy of the neuron.
		
		// If the desired accuracy has been reached, then conclude the training process of the neuron. Otherwise, continue training it.
		if (currentAccuracy[0] > neuron->stopAboveThisAccuracy) {
			prinft("\nThe accuracy of the neuron has achieved a higher one with respect to the one that was specified as a goal the very first instant it was created.\n";
			return;
		}
		
		// -------- SOLUTION OF THE REMAINING EPOCHS OF THE MODEL -------- //
		double *w_old = (double *) malloc((neuron->m+1)*neuron->p*sizeof(double)); // Allocate the memory required for the variable "w_old", which will contain the previous weight values that were obtained with respect to the current ones.
		double *errorTerm = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "errorTerm", which will contain the current error term value to be taken into consideration for the update of the weight values.
		double *errorTerm_dot_Xtilde = (double *) malloc((neuron->m+1)*neuron->p*sizeof(double)); // Allocate the memory required for the variable "errorTerm_dot_Xtilde", which will contain the resulting dot product between the error term and the transpose of "X_tilde".
		int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
		for (int currentEpoch=0; currentEpoch<(neuron->maxEpochs); currentEpoch++) {
			// Pass the data of "neuron->w_new" to "w_old".
			for (int currentCoefficient=0; currentCoefficient<(m+1); currentCoefficient++) {
				w_old[currentCoefficient] = neuron->w_new[currentCoefficient];
			}
			
			// Calculate the error term obtainable with the current weight values.
			for (int currentSample=0; currentSample<(neuron->n); currentSample++) {
				errorTerm[currentSample] = (neuron->Y[currentSample] - A_u[currentSample]) * dA_u[currentSample];
			}
			
			// We update the current weight values ("w_old") in order to obtain the new ones ("neuron->w_new").
			for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
				errorTerm_dot_Xtilde[currentRow] = 0;
				currentRowTimesN = currentRow*n;
				for (int currentSample=0; currentSample<(neuron->n); currentSample++) {
					// We first multiply all the samples of the "errorTerm" with all the samples of the transpose of "X_tilde".
					errorTerm_dot_Xtilde[currentRow] += errorTerm[currentSample] * TransposeOf_X_tilde[currentSample + currentRowTimesN];
				}
				// We now multiple the previous result with the learning rate and then update for the current weight value (which is indicated by "currentRow").
				neuron->w_new[currentRow] = w_old[currentRow] + neuron->learningRate * errorTerm_dot_Xtilde[currentRow];
			}
			
			// We recalculate the currently predicted output data made by the body of the neuron and store it in "f_x_tilde".
			for (int currentRow=0; currentRow<neuron->n; currentRow++) {
				f_x_tilde[currentRow] = neuron->w_new[0];
				currentRowTimesM = currentRow*m;
				for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
					f_x_tilde[currentRow] = f_x_tilde[currentRow] + neuron->w_new[currentColumn]*X_tilde[currentColumn + currentRowTimesM];
				}
			}
			
			// We recalculate the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
			(*activationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, neuron); // We calculate A(u) and store it in the pointer variable "A_u".
			
			// We recalculate the derivative of A(u).
		    (*derivateOfActivationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, dA_u, neuron); // We calculate the derivative of A(u) and store it in the pointer variable "dA_u".
			// NOTE: Remember that "Y_hat" = A(u) = "A_u".
			
			// We recalculate the corresponding evaluation metric with respect to the actual data of the system under study "neuron->Y" and the currently predicted output made by the neuron "A_u".
			currentAccuracy[0] = 0; // We reset the value of this variable in order to recalculate it.
			getAdjustedCoefficientOfDetermination(neuron->Y, A_u, neuron->n, neuron->m, neuron->p, 1, currentAccuracy); // We calculate the current accuracy of the neuron.
			
			// If the desired accuracy has been reached, then conclude the training process of the neuron. Otherwise, continue training it.
			if (currentAccuracy[0] > neuron->stopAboveThisAccuracy) {
				prinft("\nThe accuracy of the neuron has achieved a higher one with respect to the one that was specified as a goal when concluding the epoch number %d.\n", currentEpoch);
				return;
			}
		}
	}
	
	// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
	free(f_x_tilde);
	free(activationFunctions);
	free(A_u);
	free(derivateOfActivationFunctions);
	free(dA_u);
	free(currentAccuracy);
	free(w_old);
	free(errorTerm);
	free(errorTerm_dot_Xtilde);
	
	
	prinft("\nThe accuracy of the neuron did not surpased the defined goal but its training process has been successfully concluded.\n");
	return;
}


/**
* The "getSingleNeuronDNN()" function is used to apply the
* machine learning algorithm called simple linear regression. Within
* this process, the best fitting equation with the form of "y_hat =
* b_0 + b_1*x" will be identified with respect to the sampled data
* given through the argument pointer variables "X" and "Y". As a
* result, the identified coefficient values will be stored in the
* argument pointer variable "b".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INNITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m=1" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INNITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A SIZE OF
*					 "n" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable in which we will store
*					 the identified best fitting coefficient values
*					 for the desired machine learning algorithm. These
*					 coefficients will each be stored in the same
*					 column but under different rows where the first
*					 coefficient (b_0) will be stored in the row with
*					 index 0 and the last coefficient (b_1) will be
*					 stored in the row with index 1. IT IS
*					 INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*					 BEFORE CALLING THIS FUNCTION WITH A VARIABLE SIZE
*					 OF "m+1=2" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "b".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 28, 2021
* LAST UPDATE: N/A
*/
// ------------------ RELU ACTIVATION FUNCTION ------------------- //
static void getReluActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		if (u[currentRow] > 0) {
	        A_u[currentRow] = u[currentRow];
	    } else {
	        A_u[currentRow] = 0;
	    }
	}
}
static void getDerivateReluActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		if (u[currentRow] > 0) {
	        dA_u[currentRow] = 1;
	    } else {
	        dA_u[currentRow] = 0;
	    }
	}
}

// ------------------ TANH ACTIVATION FUNCTION ------------------- //
static void getTanhActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		A_u[currentRow] = (exp(u[currentRow]) - exp(-u[currentRow])) / (exp(u[currentRow]) + exp(-u[currentRow]));
	}
}
static void getDerivateTanhActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		dA_u[currentRow] = 1 - A_u[currentRow] * A_u[currentRow];
	}
}

// ---------- RAISE TO THE LOGISTIC POWER ACTIVATION FUNCTION --------- //
static void getLogisticActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		A_u[currentRow] = 1 / (1 + exp(-u[currentRow]));
	}
}
static void getLogisticActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		dA_u[currentRow] = A_u[currentRow] * (1 - A_u[currentRow]);
	}
}

// ---------- RAISE TO THE 1ST POWER ACTIVATION FUNCTION --------- //
static void getRaiseToTheFirstPowerActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		A_u[currentRow] = u[currentRow];
	}
}
static void getDerivateRaiseToTheFirstPowerActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		dA_u[currentRow] = 1;
	}
}

// --------------- SQUARE ROOT ACTIVATION FUNCTION --------------- //
static void getSquareRootActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		A_u[currentRow] = sqrt(u[currentRow]);
	}
}
static void getDerivateSquareRootActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		dA_u[currentRow] = 1 / (2*A_u[currentRow]);
	}
}

// --------- RAISE TO THE 2ND POWER ACTIVATION FUNCTION ---------- //
static void getRaiseToTheSecondPowerActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		A_u[currentRow] = u[currentRow] * u[currentRow];
	}
}
static void getDerivateRaiseToTheSecondPowerActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		dA_u[currentRow] = 2*u[currentRow];
	}
}

// --------- RAISE TO THE 3RD POWER ACTIVATION FUNCTION ---------- //
static void getRaiseToTheThirdPowerActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		A_u[currentRow] = u[currentRow] * u[currentRow] * u[currentRow];
	}
}
static void getDerivateRaiseToTheThirdPowerActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		dA_u[currentRow] = 3 * u[currentRow] * u[currentRow];
	}
}

// --------- RAISE TO THE 4TH POWER ACTIVATION FUNCTION ---------- //
static void getRaiseToTheFourthPowerActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
	double squareThisValue; // Variable used to store the value that wants to be squared.
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
    	squareThisValue = u[currentRow] * u[currentRow];
		A_u[currentRow] = squareThisValue * squareThisValue;
	}
}
static void getDerivateRaiseToTheFourthPowerActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		dA_u[currentRow] = 4 * u[currentRow] * u[currentRow] * u[currentRow];
	}
}

// --------- RAISE TO THE 5TH POWER ACTIVATION FUNCTION ---------- //
static void getRaiseToTheFifthPowerActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    double squareThisValue; // Variable used to store the value that wants to be squared.
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
    	squareThisValue = u[currentRow] * u[currentRow];
		A_u[currentRow] = squareThisValue * squareThisValue * u[currentRow];
	}
}
static void getDerivateRaiseToTheFifthPowerActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    double squareThisValue; // Variable used to store the value that wants to be squared.
	for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		squareThisValue = u[currentRow] * u[currentRow];
		dA_u[currentRow] = 5 * squareThisValue * squareThisValue;
	}
}

// --------- RAISE TO THE 6TH POWER ACTIVATION FUNCTION ---------- //
static void getRaiseToTheSixthPowerActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    double squareThisValue; // Variable used to store the value that wants to be squared.
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
    	squareThisValue = u[currentRow] * u[currentRow];
		A_u[currentRow] = squareThisValue * squareThisValue * squareThisValue;
	}
}
static void getDerivateRaiseToTheSixthPowerActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    double squareThisValue; // Variable used to store the value that wants to be squared.
	for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		squareThisValue = u[currentRow] * u[currentRow];
		dA_u[currentRow] = 6 * squareThisValue * squareThisValue * u[currentRow];
	}
}

// ------- 1ST ORDER DEGREE EXPONENTIAL ACTIVATION FUNCTION ------ //
static void getFirstOrderDegreeExponentialActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		A_u[currentRow] = exp(u[currentRow]);
	}
}
static void getDerivateFirstOrderDegreeExponentialActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		dA_u[currentRow] = A_u[currentRow];
	}
}

// ------- 2ND ORDER DEGREE EXPONENTIAL ACTIVATION FUNCTION ------ //
static void getSecondOrderDegreeExponentialActivation(double *u, double *A_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		A_u[currentRow] = exp(u[currentRow] * u[currentRow]);
	}
}
static void getDerivateSecondOrderDegreeExponentialActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
    for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		dA_u[currentRow] = 2 * u[currentRow] * A_u[currentRow];
	}
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
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix, containing
*				 the real/predicted results of the system under study.
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
*									INDISPENSABLE THAT THIS VARIABLE IS
*									ALLOCATED AND INNITIALIZED WITH ZERO
*									BEFORE CALLING THIS FUNCTION WITH A
*									SIZE OF "p" 'DOUBLE' MEMORY SPACES.
*									Note that the results will be stored
*									in ascending order with respect to the
*									outputs of the system under study. In
*									other words, from the first output in
*									index "0" up to the last output in
*									index "p-1".
*
* NOTE: RESULTS ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "adjustedRsquared".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 12, 2021
* LAST UPDATE: NOVEMBER 17, 2021
*/
static void getAdjustedCoefficientOfDetermination(double *realOutputMatrix, double *predictedOutputMatrix, int n, int m, int p, int degreesOfFreedom, double *adjustedRsquared) {
	// We obtain the sums required for the means to be calculated and the SSE values for each of the columns of the input matrix.
    int currentRowTimesP; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
    int currentRowAndColumn; // This variable is used to store a repetitive mathematical operations in some for-loops, for performance purposes.
    double squareThisValue; // Variable used to store the value that wants to be squared, for performance purposes.
	double *mean_realOutputMatrix = (double *) calloc(p, sizeof(double)); // This pointer variable is used to store the means of all the outputs of the argument pointer variable "realOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesP = currentRow*p;
    	for (int currentOutput=0; currentOutput<p; currentOutput++) {
    		// We make the required calculations to obtain the SSE values.
			currentRowAndColumn = currentOutput + currentRowTimesP;
			squareThisValue = realOutputMatrix[currentRowAndColumn] - predictedOutputMatrix[currentRowAndColumn];
			adjustedRsquared[currentOutput] += (squareThisValue * squareThisValue); // We temporarly store the SSE values in the argument pointer variable "adjustedRsquared", for performance purposes.
			
			// We make the required calculations to obtain the sums required for the calculation of the means.
    		mean_realOutputMatrix[currentOutput] += realOutputMatrix[currentRowAndColumn];
		}
	}
	// We apply the final operations required to complete the calculation of the means.
	for (int currentOutput=0; currentOutput<p; currentOutput++) {
		mean_realOutputMatrix[currentOutput] = mean_realOutputMatrix[currentOutput]/n;
	}
	
	// We obtain the SST values that will be required to make the calculation of the adjusted coefficient of determination.
	double *SST = (double *) calloc(p, sizeof(double)); // This pointer variable is used to store the SST values for all the outputs of the argument pointer variable "realOutputMatrix".
	for (int currentRow = 0; currentRow < n; currentRow++) {
		currentRowTimesP = currentRow*p;
    	for (int currentOutput=0; currentOutput<p; currentOutput++) {
    		// We make the required calculations to obtain the SST values.
			squareThisValue = realOutputMatrix[currentOutput + currentRowTimesP] - mean_realOutputMatrix[currentOutput];
			SST[currentOutput] += (squareThisValue * squareThisValue); // We temporarly store the SSE values in the argument pointer variable "R", for performance purposes.
		}
	}
	
	// Finally, we calculate the adjusted coefficient of determination and store its results in the pointer variable "adjustedRsquared".
	for (int currentOutput=0; currentOutput<p; currentOutput++) {
		adjustedRsquared[currentOutput] = 1 - ( (adjustedRsquared[currentOutput]/(n-m-degreesOfFreedom))/(SST[currentOutput]/(n-degreesOfFreedom)) );
	}
	
	// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
	free(mean_realOutputMatrix);
	free(SST);
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
static void getAccuracy(double *realOutputMatrix, double *predictedOutputMatrix, int n, int p, double *accuracy) {
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
* The "predictSingleNeuronDNN()" function is used to make the
* predictions of the requested input values (X) by applying the
* simple linear equation system with the specified coefficient values
* (b). The predicted values will be stored in the argument pointer
* variable "Y_hat".
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning predictions will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INNITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m=1" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *b - This argument will contain the pointer to a
*					 memory allocated variable containing the
*					 coefficient values for the desired machine
*					 learning algorithm and that will be used to make
*					 the specified predictions. IT IS INDISPENSABLE
*					 THAT THIS VARIABLE IS ALLOCATED AND INNITIALIZED
*					 BEFORE CALLING THIS FUNCTION WITH A VARIABLE
*					 SIZE OF "m+1=2" 'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
*
* @param int m - This argument will represent the total number of
*				 features (independent variables) that the input
*				 matrix has, with which the output data was obtained.
*
* @param int p - This argument will represent the total number of 
*				 outputs that exist in the the output matrix,
*				 containing the real results of the system under
*				 study.
*
* @param double *Y_hat - This argument will contain the pointer to a
*					 	 memory allocated output matrix, representing
*					 	 the predicted data of the system under study.
*						 THIS VARIABLE SHOULD BE ALLOCATED BEFORE
*						 CALLING THIS FUNCTION WITH A SIZE OF "n"
*						 TIMES "p=1" 'DOUBLE' MEMORY SPACES. The
*						 results will be stored in the same order as
*						 the input data given such that the first
*						 sample will be stored in the row with index
*						 "0" and the last sample in the row with
*						 index "n".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER XX, 2021
* LAST UPDATE: N/A
*/
void predictSingleNeuronDNN(struct singleNeuronDnnStruct *neuron, double *Y_hat) {
	// If the machine learning features exceed the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (m != 1) {
		printf("\nERROR: The machine learning features (independent variables) must be equal to 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study exceed the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (p != 1) {
		printf("\nERROR: The outputs of the system under study must be equal to 1 for this particular algorithm.\n");
		exit(1);
	}
	
	// We predict all the requested input values (X) with the desired machine learning algorithm and its especified coefficient values (b).
	for (int currentRow = 0; currentRow < n; currentRow++) {
		Y_hat[currentRow] = b[0] + b[1]*X[currentRow];
	}
}

