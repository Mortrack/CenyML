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
* The "getSingleNeuronDNN()" function is used to apply the machine
* learning algorithm called single neuron in Deep Neural Network as
* formulated in the master thesis of Cesar Miranda Meza called
* "Machine learning to support applications with embedded systems
* and parallel computing". Within this process, the best fitting
* equation with the form of "y_hat = b_0 + w_1*x_1 + w_2*x_2
* + ... + w_m*x_m" will be identified with respect to the sampled
* data given through the argument pointer variables "X" and "Y". As
* a result, the identified coefficient values will be stored in the
* argument pointer variable "neuron->w_new". With respect to the
* struct pointer variable "neuron", it should contain all the
* information required in order to be able to create and make an
* artificial neuron. Its accessible inner elements will be described
* in the following list:
* 
* @param double *X - This argument will contain the pointer to a
*					 memory allocated input matrix, from which the
*					 desired machine learning algorithm will be
*					 calculated. THIS VARIABLE SHOULD BE ALLOCATED
*					 AND INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "m" 'DOUBLE' MEMORY
*					 SPACES.
*
* @param double *w_first - This argument will contain the pointer to
*						   a memory allocated coefficient matrix.
*						   The use of this variable will difer
*						   depending on the value assigned in the
*						   argument variable "isInitial_w", whose
*						   possible outcomes are listed below:
*						   1) "isInitial_w"=(int)1 --> "w_first"
*							  HAS TO BE INITIALIZED BEFORE CALLING
*							  THIS FUNCTION because its defined
*							  coefficient values will be assigned
*							  to the neuron as its initial weight
*							  values before starting its training
*							  process.
*						   2) "isInitial_w"=(int)0 --> "w_first"
*							  does not require to be initialized but
*							  has to be allocated in memory. After
*							  this function concludes its processes,
*							  the implementer will be able to know
*							  what were the initial weight values
*							  that the neuron had when it was
*							  created.
*						   Regardless of the value of "isInitial_w",
*						   "w_first" SHOULD BE ALLOCATED BEFORE
*						   CALLING THIS FUNCTION WITH A SIZE OF "1"
*						   TIMES "m+1" 'DOUBLE' MEMORY SPACES.
*
* @param double *Y - This argument will contain the pointer to a
*					 memory allocated output matrix, representing
*					 the real data of the system under study. This
*					 variable will be used as a reference to apply
*					 the desired machine learning algorithm. THIS
*					 VARIABLE SHOULD BE ALLOCATED AND INITIALIZED
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
* @param char isInitial_w = This argument variable will work as a flag
*							to indicate whether the coefficients
*							contained in the argument variable
*							"w_first" will be used as the initial
*							weight values for the neuron to be created
*							or not. The possible values for
*							"isInitial_w" are the following:
*				  		    1) "isInitial_w"=(int)1 --> the coefficient
*								values of "w_first" will be assigned
*							    to the neuron as its initial weight
*							    values before starting its training
*							    process.
*						    2) "isInitial_w"=(int)0 --> the coefficient
*								values of "w_first" will not be
*								assigned to the neuron as its initial
*								weight values and after having called
*								this function, the implementer will be
*								able to retrieve from "w_first" the
*								coefficient values with which the neuron
*								had been created before starting its
*								learning process.
*
* @param char isClassification = This argument variable will work as a
*						  		 flag to indicate to the neron if it is
*								 expected from it to learn from the given
*								 data of "X" and "Y" as if their were
*								 meant for a classification problem or not.
*								 The possible valid values for this flag
*								 are the following:
*					  		     1) "isClassification" = (int) 1 --> The
*									neuron will interpret the data of "X"
*									and "Y" as if it were meant for a
*									classification problem.
*								 2) "isClassification" = (int) 0 --> The
*									neuron will interpret the data of "X"
*									and "Y" as if it were meant for a
*									regression problem.
*
* @param double threshold - This argument will represent desired threshold
*							that the implementer desired the neuron to
*							consider in classification problems. In this
*							regard, whenever the predicted output of the
*							neuron is higher than the defined threshold
*							value, then that prediction should be
*							interpreted as group 1 (ussually refered to as
*							the binary output 1). Conversely, if the
*							predicted value is lower than the defined
*							threshold value, then that prediction should be
*							interpreted as group 2 (ussually refered to as
*							the binary output 0). However, have in mind that
*							"threshold" will only be used by the neuron if
*							the argument variable "isClassification" = 1.
*
* @param int desiredValueForGroup1 - This argument will represent the desired
*									 label value to whenever an output of the
*									 neuron predicts the classification group
*									 1. Ussually, this is label with the value
*									 of "(int) 1" but any other customized
*									 value can be assigned by the implementer.
*									 However, have in mind that this argument
*									 variable will be considered by the neuron
*									 as long as the argument variable
*									 "isClassification" = 1 and only when the
*									 implementer requests to the neuron a
*									 prediction through the function
*									 "predictSingleNeuronDNN()".
*
* @param int desiredValueForGroup2 - This argument will represent the desired
*									 label value to whenever an output of the
*									 neuron predicts the classification group
*									 2. Ussually, this is label with the value
*									 of "(int) -1" but any other customized
*									 value can be assigned by the implementer.
*									 However, have in mind that this argument
*									 variable will be considered by the neuron
*									 as long as the argument variable
*									 "isClassification" = 1 and only when the
*									 implementer requests to the neuron a
*									 prediction through the function
*									 "predictSingleNeuronDNN()".
*
* @param int activationFunctionToBeUsed - This argument will represent the
*										  identifier of the desired activation
*										  function ot be used by the neuron
*										  during its training process. Its
*										  possible valid values are the
*										  following:
*										  0 = Rectified Linear Units (ReLU).
*										  1 = Hyperbolic tangent (tanh).
*										  2 = Logistic function.
*										  3 = Raise to the 1st power.
*										  4 = Raise to the 2nd power.
*										  5 = Raise to the 3rd power.
*										  6 = Raise to the 4th power.
*										  7 = Raise to the 5th power.
*										  8 = Raise to the 6th power.
*										  9 = 1st order degree exponential.
*										  10 = 2nd order degree exponential.
*
* @param double learningRate - This argument will represent hyperparameter
*							   value to be used by the learning rate of the
*							   artificial neuron to be created. Note that there
*							   is no way to know what is going to be the best
*							   learning rate value for your particular problem
*							   because the best one differs from one problem to
*							   another. Therefore, you will most likely have to
*							   experiment with several values until you find
*							   the model solution that satisfies you the most.
*
* @param double stopAboveThisAccuracy - This argument will represent a a stop
*										value for the training process. The way
*										this value will work is that if the
*										neuron gets an evaluation metric result
*										that is strictly higher than the one
*										defined in "stopAboveThisAccuracy", then
*										the neuron will stop its training process
*										and the function "getSingleNeuronDNN()"
*										will end. Note that if "isClassification"
*										= (int) 1, then the evaluation metric to
*										be used will be the accuracy (for
*										classification). Conversely, if
*										"isClassification" = (int) 0, then the
*										evaluation metric to be used will be the
*										adjusted R squared (for regression).
*
* @param int maxEpochs - This argument will represent the maximum number of
*						 epochs that are desired for the training process of the
*						 artificial neuron. Note that for each epoch that occurs,
*						 that should be interpreted as the neuron having updated
*						 its weight values one time.
*
* @param char isReportLearningProgress = This argument variable will work as a
*						  		 		 flag to indicate to the neuron if it is
*								 		 desired that it reports its learning
*										 progress to the user. The following
*										 will list the possible valid outcomes
*										 for this variable:
*					  		     		 1) "isReportLearningProgress" = (int) 1:
*										     The neuron will interpret this as
*											 being instructed to report its
*											 learning progress to the user
*											 through the window terminal by
*											 displaying messages over time.
*								 		 2) "isReportLearningProgress" = (int) 0:
*										     The neuron will interpret this as
*											 being instructed not to report its
*											 learning progress.
*
* @param int reportEachSpecifiedEpochs - This argument variable will indicate
*										 how many each amount of epochs it is
*										 desired by the implementer that the
*										 artificial neuron reports its learning
*										 progress to the user. However, in order
*										 for the neuron to consider this variable,
*										 it will be strictly needed to set the
*										 argument variable
*										 "isReportLearningProgress" = (int) 1.
*
* @param double *w_best - This argument will contain the pointer to a memory
*						 allocated variable in which we will store the
*						 identified best fitting coefficient values for the
*						 model of a single neuron in Deep Neural Network. These
*						 coefficients will each be stored in the same row but
*						 under different columns where the first coefficient
*						 (b_0) will be stored in the column with index 0; the
*						 second coefficient (w_1) will be stored in the column
*						 index 1 and; the last coefficient (w_m) will be stored
*						 in the column index m. IT IS INDISPENSABLE THAT THIS
*						 VARIABLE IS ALLOCATED BEFORE CALLING THIS FUNCTION
*						 WITH A VARIABLE SIZE OF "1" TIMES "m+1" 'DOUBLE'
*						 MEMORY SPACES.
*
* @param double bestAccuracy - This argument will contain the value of the best
*							   accuracy that the neuron was able to achieve
*							   during its training process.
*
* @param double *w_new - This argument will contain the pointer to a memory
*						 allocated variable in which we will store the last
*						 identified coefficient values for the model of a
*						 single neuron in Deep Neural Network. These
*						 coefficients will each be stored in the same row but
*						 under different columns where the first coefficient
*						 (b_0) will be stored in the column with index 0; the
*						 second coefficient (w_1) will be stored in the column
*						 index 1 and; the last coefficient (w_m) will be stored
*						 in the column index m. IT IS INDISPENSABLE THAT THIS
*						 VARIABLE IS ALLOCATED BEFORE CALLING THIS FUNCTION
*						 WITH A VARIABLE SIZE OF "1" TIMES "m+1" 'DOUBLE'
*						 MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"w_new" that is contained in the struct pointer variable
*		"neuron".
*
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 29, 2021
* LAST UPDATE: JANUARY 09, 2022
*/
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
	// We obtain and store the transpose of "X_tilde".
	int currentRowTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int mPlusOne = neuron->m+1; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	double *TransposeOf_X_tilde = (double *) malloc(mPlusOne*neuron->n*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = 0; // This variable is used in the for-loop for the matrix transpose that will be made.
	for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		currentRow2 = 0; // We reset the counters used in the following for-loop.
		currentRowTimesMplusOne = currentRow*mPlusOne;
		currentRowTimesM = currentRow*neuron->m;
		TransposeOf_X_tilde[currentColumn2] = 1;
		for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
			currentRow2++;
			TransposeOf_X_tilde[currentColumn2 + currentRow2*neuron->n] = neuron->X[currentColumn-1 + currentRowTimesM];
		}
		currentColumn2++;
	}
	
	// -------------------- WEIGHT INITIALIZATION -------------------- //
	// Store the initial weight values into "neuron->w_new" and into "neuron->w_best".
	if (neuron->isInitial_w == 0) {
		// In order to initialize "neuron->w_new" with random values between "-1" and "+1"., intialize random number generator.
	    time_t t;
		srand((unsigned) time(&t));
	    
	    // Initialize "neuron->w_new" with random values between -1 to +1 with three decimals at the most. Give the save values to "neuron->w_best".
	    double currentRandomNumber;
	    for (int current_w=0 ; current_w<(neuron->m+1); current_w++) {
	        currentRandomNumber = ((float) (rand() % 1000))/500 - 1;
	        neuron->w_first[current_w] = currentRandomNumber;
	        neuron->w_new[current_w] = currentRandomNumber;
	        neuron->w_best[current_w] = currentRandomNumber;
	    }
	} else if (neuron->isInitial_w == 1) {
		// Pass the values of "neuron->w_first" to "neuron->w_new" and "neuron->w_best".
		for (int current_w=0 ; current_w<(neuron->m+1); current_w++) {
	        neuron->w_new[current_w] = neuron->w_first[current_w];
	        neuron->w_best[current_w] = neuron->w_first[current_w];
	    }
	}
	
	// We allocate all the memory that will be required for the training process of the neuron.
	double *f_x_tilde = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "f_x_tilde", which will contain the currently predicted output data made by the body of the neuron.
	// NOTE: "activationFunctions" is a pointer to each of the individual activation functions that were developed as static void functions.
    static void (*activationFunctions[])(double *, double *, struct singleNeuronDnnStruct *) = {getReluActivation, getTanhActivation, getLogisticActivation, getRaiseToTheFirstPowerActivation, getRaiseToTheSecondPowerActivation, getRaiseToTheThirdPowerActivation, getRaiseToTheFourthPowerActivation, getRaiseToTheFifthPowerActivation, getRaiseToTheSixthPowerActivation, getFirstOrderDegreeExponentialActivation, getSecondOrderDegreeExponentialActivation};
    double *A_u = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "A_u", which will contain the currently predicted output data made by the neuron.
    // NOTE: "derivateOfActivationFunctions" is a pointer to each of the individual derivatives of the activation functions that were developed as static void functions.
	static void (*derivateOfActivationFunctions[])(double *, double *, double *, struct singleNeuronDnnStruct *) = {getDerivateReluActivation, getDerivateTanhActivation, getDerivateLogisticActivation, getDerivateRaiseToTheFirstPowerActivation, getDerivateRaiseToTheSecondPowerActivation, getDerivateRaiseToTheThirdPowerActivation, getDerivateRaiseToTheFourthPowerActivation, getDerivateRaiseToTheFifthPowerActivation, getDerivateRaiseToTheSixthPowerActivation, getDerivateFirstOrderDegreeExponentialActivation, getDerivateSecondOrderDegreeExponentialActivation};
    double *dA_u = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "dA_u", which will contain the derivative of A(u).
    double currentAccuracy = 0; // Declare the variable "currentAccuracy", which will contain the current accuracy of the neuron.
    double *w_old = (double *) malloc((neuron->m+1)*neuron->p*sizeof(double)); // Allocate the memory required for the variable "w_old", which will contain the previous weight values that were obtained with respect to the current ones.
	double *errorTerm = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "errorTerm", which will contain the current error term value to be taken into consideration for the update of the weight values.
	double *errorTerm_dot_Xtilde = (double *) malloc((neuron->m+1)*neuron->p*sizeof(double)); // Allocate the memory required for the variable "errorTerm_dot_Xtilde", which will contain the resulting dot product between the error term and the transpose of "X_tilde".
	int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	
	// Determine if the requested model to generate is meant for a classification or for a regression problem to then solve it accordingly.
	if (neuron->isClassification == 1) {
		// ----------------------------------------- //
		// ----- CLASSIFICATION MODEL SELECTED ----- //
		// ----------------------------------------- //
		
		
		// ----------- ADDITIONAL PREPROCESSING NEEDED WHEN A CLASSIFICATION MODEL IS REQUESTED ----------- //
		// We temporarily manage group 1 with output values of "1" and the group 2 with output values of "0" so that the accuracy can be properly measured by the neuron.
		for (int currentRow=0; currentRow<neuron->n; currentRow++) {
			if (neuron->Y[currentRow] == neuron->desiredValueForGroup1) {
				neuron->Y[currentRow] = 1;
			} else if (neuron->Y[currentRow] == neuron->desiredValueForGroup2) {
				neuron->Y[currentRow] = 0;
			} else {
				printf("\nERROR: From the singleNeuronDnnStruct structure variable that was used to train a deep learning model, the value from the row %d of the allocated variable \"Y\" did not matched any of the specified values in the variables \"desiredValueForGroup1\" and \"desiredValueForGroup2\".\n", currentRow);
				exit(1);
			}
		}
		
		// ----------- EVALUATION OF THE INITIAL WEIGHT VALUES ----------- //
		// We calculate the currently predicted output data made by the body of the neuron and store it in "f_x_tilde".
		for (int currentRow=0; currentRow<neuron->n; currentRow++) {
			f_x_tilde[currentRow] = neuron->w_new[0];
			currentRowTimesM = currentRow*neuron->m;
			for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
				f_x_tilde[currentRow] = f_x_tilde[currentRow] + neuron->w_new[currentColumn]*neuron->X[currentColumn-1 + currentRowTimesM];
			}
		}
		
		// We calculate, in its continous (regression) form, the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
		(*activationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, neuron); // We calculate A(u) and store it in the pointer variable "A_u".
		
		// We calculate the derivative of A(u).
		// NOTE: Remember that "Y_hat" = A(u) = "A_u".
	    (*derivateOfActivationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, dA_u, neuron); // We calculate the derivative of A(u) and store it in the pointer variable "dA_u".
		
		
		// We apply the threshold define by the implementer in order to obtain a classification output and store it in "A_u".
		for (int currentRow=0; currentRow<neuron->n; currentRow++) {
			if (A_u[currentRow] > neuron->threshold) { // For performance purposes and compatibility with the accuracy method to be used, the classification output results will be either 1 or 0.
				A_u[currentRow] = 1;
			} else {
				A_u[currentRow] = 0;
			}
		}
		
		// We calculate the corresponding evaluation metric with respect to the actual data of the system under study "neuron->Y" and the currently predicted output made by the neuron "A_u".
		getNeuronAccuracy(neuron->Y, A_u, neuron->n, &currentAccuracy); // We calculate the current accuracy of the neuron.
		neuron->bestAccuracy = currentAccuracy; // We pass the current accuracy to the best accuracy record because this is the evaluation of the very first weight values.
		
		// If the desired accuracy has been reached, then conclude the training process of the neuron. Otherwise, continue training it.
		if (currentAccuracy > neuron->stopAboveThisAccuracy) {
			printf("\nThe accuracy (%f) of the neuron has achieved a higher one with respect to the one that was specified as a goal the very first instant it was created.\n", currentAccuracy);
			
			// ----------- ADDITIONAL POSTPROCESSING NEEDED WHEN A CLASSIFICATION MODEL IS REQUESTED ----------- //
			// We restore the original output values defined for group 1 and group 2.
			for (int currentRow=0; currentRow<neuron->n; currentRow++) {
				if (neuron->Y[currentRow] == 1) {
					neuron->Y[currentRow] = neuron->desiredValueForGroup1;
				} else {
					neuron->Y[currentRow] = neuron->desiredValueForGroup2;
				}
			}
		
			// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
			free(TransposeOf_X_tilde);
			free(f_x_tilde);
			free(A_u);
			free(dA_u);
			free(w_old);
			free(errorTerm);
			free(errorTerm_dot_Xtilde);
			return;
		}
		
		// -------- BEGINNING OF THE EPOCHS OF THE MODEL ------- //
		for (int currentEpoch=0; currentEpoch<(neuron->maxEpochs); currentEpoch++) {
			// Pass the data of "neuron->w_new" to "w_old".
			for (int currentCoefficient=0; currentCoefficient<(neuron->m+1); currentCoefficient++) {
				w_old[currentCoefficient] = neuron->w_new[currentCoefficient];
			}
			
			// Calculate the error term obtainable with the current weight values.
			for (int currentSample=0; currentSample<(neuron->n); currentSample++) {
				errorTerm[currentSample] = (neuron->Y[currentSample] - A_u[currentSample]) * dA_u[currentSample];
			}
			
			// We update the current weight values ("w_old") in order to obtain the new ones ("neuron->w_new").
			for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
				errorTerm_dot_Xtilde[currentRow] = 0;
				currentRowTimesN = currentRow*neuron->n;
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
				currentRowTimesM = currentRow*neuron->m;
				for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
					f_x_tilde[currentRow] = f_x_tilde[currentRow] + neuron->w_new[currentColumn]*neuron->X[currentColumn-1 + currentRowTimesM];
				}
			}
			
			// We recalculate the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
			(*activationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, neuron); // We calculate A(u) and store it in the pointer variable "A_u".
			
			// We recalculate the derivative of A(u).
			// NOTE: Remember that "Y_hat" = A(u) = "A_u".
		    (*derivateOfActivationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, dA_u, neuron); // We calculate the derivative of A(u) and store it in the pointer variable "dA_u".
			
			// We apply the threshold define by the implementer in order to obtain a classification output and store it in "A_u".
			for (int currentRow=0; currentRow<neuron->n; currentRow++) {
				if (A_u[currentRow] > neuron->threshold) { // For performance purposes and compatibility with the accuracy method to be used, the classification output results will be either 1 or 0.
					A_u[currentRow] = 1;
				} else {
					A_u[currentRow] = 0;
				}
			}
		
			// We recalculate the corresponding evaluation metric with respect to the actual data of the system under study "neuron->Y" and the currently predicted output made by the neuron "A_u".
			currentAccuracy = 0; // We reset the value of this variable in order to recalculate it.
			getNeuronAccuracy(neuron->Y, A_u, neuron->n, &currentAccuracy); // We calculate the current accuracy of the neuron.
			
			// We compare the accuracy of the currently obtained weight values with respect to the latest best one recorded. If the current one is better than the recorded one, then store the current one in its place and do the same for the best recorded weight values.
			if ((currentAccuracy) > (neuron->bestAccuracy)) {
				neuron->bestAccuracy = currentAccuracy; // Pass the value of the current accuracy into "neuron->bestAccuracy".
				for (int current_w=0 ; current_w<(neuron->m+1); current_w++) { // Pass the values of "neuron->w_new" to "neuron->w_best".
			        neuron->w_best[current_w] = neuron->w_new[current_w];
			    }
			}
			
			// Determine whether it was requested that the neuron reports its learning progress or not.
			if (neuron->isReportLearningProgress == 1) { // If the implementer requested the neuron to report its progress, apply the following code.
				if ((currentEpoch % neuron->reportEachSpecifiedEpochs) == 0) { // Make neuron report at each "neuron->reportEachSpecifiedEpochs" epochs.
		            printf("\nEpoch %d --> single neuron in DNN has achieved an accuracy of %f\n", currentEpoch+1, currentAccuracy);
		        }
			}
			
			// If the desired accuracy has been reached, then conclude the training process of the neuron. Otherwise, continue training it.
			if (currentAccuracy > neuron->stopAboveThisAccuracy) {
				printf("\nThe accuracy (%f) of the neuron has achieved a higher one with respect to the one that was specified as a goal when concluding the epoch number %d.\n", currentAccuracy, currentEpoch+1);
				
				// ----------- ADDITIONAL POSTPROCESSING NEEDED WHEN A CLASSIFICATION MODEL IS REQUESTED ----------- //
				// We restore the original output values defined for group 1 and group 2.
				for (int currentRow=0; currentRow<neuron->n; currentRow++) {
					if (neuron->Y[currentRow] == 1) {
						neuron->Y[currentRow] = neuron->desiredValueForGroup1;
					} else {
						neuron->Y[currentRow] = neuron->desiredValueForGroup2;
					}
				}
		
				// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
				free(TransposeOf_X_tilde);
				free(f_x_tilde);
				free(A_u);
				free(dA_u);
				free(w_old);
				free(errorTerm);
				free(errorTerm_dot_Xtilde);
				return;
			}
		}
		
		// ----------- ADDITIONAL POSTPROCESSING NEEDED WHEN A CLASSIFICATION MODEL IS REQUESTED ----------- //
		// We restore the original output values defined for group 1 and group 2.
		for (int currentRow=0; currentRow<neuron->n; currentRow++) {
			if (neuron->Y[currentRow] == 1) {
				neuron->Y[currentRow] = neuron->desiredValueForGroup1;
			} else {
				neuron->Y[currentRow] = neuron->desiredValueForGroup2;
			}
		}
	} else {
		// ------------------------------------- //
		// ----- REGRESSION MODEL SELECTED ----- //
		// ------------------------------------- //
		
		
		// ----------- EVALUATION OF THE INITIAL WEIGHT VALUES ----------- //
		// We calculate the currently predicted output data made by the body of the neuron and store it in "f_x_tilde".
		for (int currentRow=0; currentRow<neuron->n; currentRow++) {
			f_x_tilde[currentRow] = neuron->w_new[0];
			currentRowTimesM = currentRow*neuron->m;
			for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
				f_x_tilde[currentRow] = f_x_tilde[currentRow] + neuron->w_new[currentColumn]*neuron->X[currentColumn-1 + currentRowTimesM];
			}
		}
		
		// We calculate the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
		(*activationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, neuron); // We calculate A(u) and store it in the pointer variable "A_u".
		
		// We calculate the derivative of A(u).
	    // NOTE: Remember that "Y_hat" = A(u) = "A_u".
		(*derivateOfActivationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, dA_u, neuron); // We calculate the derivative of A(u) and store it in the pointer variable "dA_u".
		
		
		// We calculate the corresponding evaluation metric with respect to the actual data of the system under study "neuron->Y" and the currently predicted output made by the neuron "A_u".
		getNeuronAdjustedCoefficientOfDetermination(neuron->Y, A_u, neuron->n, neuron->m, 1, &currentAccuracy); // We calculate the current accuracy of the neuron.
		neuron->bestAccuracy = currentAccuracy; // We pass the current accuracy to the best accuracy record because this is the evaluation of the very first weight values.
		
		// If the desired accuracy has been reached, then conclude the training process of the neuron. Otherwise, continue training it.
		if (currentAccuracy > neuron->stopAboveThisAccuracy) {
			printf("\nThe adjusted R squared (%f) of the neuron has achieved a higher one with respect to the one that was specified as a goal the very first instant it was created.\n", currentAccuracy);
			
			// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
			free(TransposeOf_X_tilde);
			free(f_x_tilde);
			free(A_u);
			free(dA_u);
			free(w_old);
			free(errorTerm);
			free(errorTerm_dot_Xtilde);
			return;
		}
		
		// -------- BEGINNING OF THE EPOCHS OF THE MODEL ------- //
		for (int currentEpoch=0; currentEpoch<(neuron->maxEpochs); currentEpoch++) {
			// Pass the data of "neuron->w_new" to "w_old".
			for (int currentCoefficient=0; currentCoefficient<(neuron->m+1); currentCoefficient++) {
				w_old[currentCoefficient] = neuron->w_new[currentCoefficient];
			}
			
			// Calculate the error term obtainable with the current weight values.
			for (int currentSample=0; currentSample<(neuron->n); currentSample++) {
				errorTerm[currentSample] = (neuron->Y[currentSample] - A_u[currentSample]) * dA_u[currentSample];
			}
			
			// We update the current weight values ("w_old") in order to obtain the new ones ("neuron->w_new").
			for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
				errorTerm_dot_Xtilde[currentRow] = 0;
				currentRowTimesN = currentRow*neuron->n;
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
				currentRowTimesM = currentRow*neuron->m;
				for (int currentColumn=1; currentColumn<mPlusOne; currentColumn++) {
					f_x_tilde[currentRow] = f_x_tilde[currentRow] + neuron->w_new[currentColumn]*neuron->X[currentColumn-1 + currentRowTimesM];
				}
			}
			
			// We recalculate the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
			(*activationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, neuron); // We calculate A(u) and store it in the pointer variable "A_u".
			
			// We recalculate the derivative of A(u).
			// NOTE: Remember that "Y_hat" = A(u) = "A_u".
		    (*derivateOfActivationFunctions[neuron->activationFunctionToBeUsed])(f_x_tilde, A_u, dA_u, neuron); // We calculate the derivative of A(u) and store it in the pointer variable "dA_u".
			
			// We recalculate the corresponding evaluation metric with respect to the actual data of the system under study "neuron->Y" and the currently predicted output made by the neuron "A_u".
			currentAccuracy = 0; // We reset the value of this variable in order to recalculate it.
			getNeuronAdjustedCoefficientOfDetermination(neuron->Y, A_u, neuron->n, neuron->m, 1, &currentAccuracy); // We calculate the current accuracy of the neuron.
			
			// We compare the accuracy of the currently obtained weight values with respect to the latest best one recorded. If the current one is better than the recorded one, then store the current one in its place and do the same for the best recorded weight values.
			if ((currentAccuracy) > (neuron->bestAccuracy)) {
				neuron->bestAccuracy = currentAccuracy; // Pass the value of the current accuracy into "neuron->bestAccuracy".
				for (int current_w=0 ; current_w<(neuron->m+1); current_w++) { // Pass the values of "neuron->w_new" to "neuron->w_best".
			        neuron->w_best[current_w] = neuron->w_new[current_w];
			    }
			}
			
			// Determine whether it was requested that the neuron reports its learning progress or not.
			if (neuron->isReportLearningProgress == 1) { // If the implementer requested the neuron to report its progress, apply the following code.
				if ((currentEpoch % neuron->reportEachSpecifiedEpochs) == 0) { // Make neuron report at each "neuron->reportEachSpecifiedEpochs" epochs.
		            printf("\nEpoch %d --> single neuron in DNN has achieved an adjusted R squared of %f\n", currentEpoch+1, currentAccuracy);
		        }
			}
			
			// If the desired accuracy has been reached, then conclude the training process of the neuron. Otherwise, continue training it.
			if (currentAccuracy > neuron->stopAboveThisAccuracy) {
				printf("\nThe adjusted R squared (%f) of the neuron has achieved a higher one with respect to the one that was specified as a goal when concluding the epoch number %d.\n", currentAccuracy, currentEpoch+1);
				
				// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
				free(TransposeOf_X_tilde);
				free(f_x_tilde);
				free(A_u);
				free(dA_u);
				free(w_old);
				free(errorTerm);
				free(errorTerm_dot_Xtilde);
				return;
			}
		}
	}
	
	// Determine whether it was requested that the neuron reports its learning progress or not.
	if (neuron->isReportLearningProgress == 1) { // If the implementer requested the neuron to report its progress, apply the following code.
		// Make the neuron report its last progress made.
		if (neuron->isClassification == 1) {
			printf("\nEpoch %d --> single neuron in DNN has achieved an accuracy of %f\n", neuron->maxEpochs, currentAccuracy);
		} else {
			printf("\nEpoch %d --> single neuron in DNN has achieved an adjusted R squared of %f\n", neuron->maxEpochs, currentAccuracy);
		}
	}
	
	// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
	free(TransposeOf_X_tilde);
	free(f_x_tilde);
	free(A_u);
	free(dA_u);
	free(w_old);
	free(errorTerm);
	free(errorTerm_dot_Xtilde);
	
	if (neuron->isClassification == 1) {
		printf("\nThe best accuracy (%f) achieved by the neuron did not surpased the defined goal but its training process has been successfully concluded.\n", neuron->bestAccuracy);
	} else {
		printf("\nThe best adjusted R squared (%f) achieved by the neuron did not surpased the defined goal but its training process has been successfully concluded.\n", neuron->bestAccuracy);
	}
	return;
}


/**
* The following static functions have the purpose of applying the
* requested activation function and/or derivative of such activation
* function by the callable functions: "getSingleNeuronDNN()" and
* "predictSingleNeuronDNN()". In this regard, the list of all the
* static functions that will apply an activation function, are the
* following:
*
* 1) getReluActivation() --> Applies the ReLU activation function.
* 2) getTanhActivation() --> Applies the tanh activation function.
* 3) getLogisticActivation() --> Applies the Logistic activation function.
* 4) getRaiseToTheFirstPowerActivation() --> Applies the raise to the 1st power activation function.
* 5) getRaiseToTheSecondPowerActivation() --> Applies the raise to the 2nd power activation function.
* 6) getRaiseToTheThirdPowerActivation() --> Applies the raise to the 3rd power activation function.
* 7) getRaiseToTheFourthPowerActivation() --> Applies the raise to the 4th power activation function.
* 8) getRaiseToTheFifthPowerActivation() --> Applies the raise to the 5th power activation function.
* 9) getRaiseToTheSixthPowerActivation() --> Applies the raise to the 6th power activation function.
* 10) getFirstOrderDegreeExponentialActivation() --> Applies the 1st order degree exponential activation function.
* 11) getSecondOrderDegreeExponentialActivation() --> Applies the 2nd order degree exponential activation function.
* For all these functions that apply a derivate, the following will
* explain how to use their argument variables and what considerations
* must have be taken into account:
*
* @param double *u - This argument will contain the pointer to a
*					 memory allocated input matrix, in which the
*					 output of the body of a neuron should be
*					 stored. THIS VARIABLE SHOULD BE ALLOCATED AND
*					 INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "p=1" 'DOUBLE'
*					 MEMORY SPACES.
*
* @param double *A_u - This argument will contain the pointer to a
*					 memory allocated output matrix in which any of
*					 these functions will store the result of
*					 applying the requested activation function on
*					 the pointer argument variable "u". "A_u"
*					 SHOULD BE ALLOCATED BEFORE CALLING THIS
*					 FUNCTION WITH A SIZE OF "n" TIMES "p=1"
*					 'DOUBLE' MEMORY SPACES.
*
* @param struct singleNeuronDnnStruct *neuron - This argument will
*					 contain the pointer to a struct variable that
*					 should contain all the information required in
*					 order to be able to create and make an
*					 artificial neuron. Its accessible inner
*					 elements are described in the list showed in
*					 the commented documentation of the function
*					 "getSingleNeuronDNN()".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"A_u".
* 
* @return void
*
* ------------------------------------------------------------------
* ------------------------------------------------------------------
*
* On the other hand, the list of all the static functions that will
* apply the derivative of such activation functions, are the following:
*
* 1) getDerivateReluActivation() --> Derivative of ReLU activation function.
* 2) getDerivateTanhActivation() --> Derivative of tanh activation function.
* 3) getDerivateLogisticActivation() --> Derivative of Logistic activation function.
* 4) getDerivateRaiseToTheFirstPowerActivation() --> Derivative of raise to the 1st power activation function.
* 5) getDerivateRaiseToTheSecondPowerActivation() --> Derivative of raise to the 2nd power activation function.
* 6) getDerivateRaiseToTheThirdPowerActivation() --> Derivative of raise to the 3rd power activation function.
* 7) getDerivateRaiseToTheFourthPowerActivation() --> Derivative of raise to the 4th power activation function.
* 8) getDerivateRaiseToTheFifthPowerActivation() --> Derivative of raise to the 5th power activation function.
* 9) getDerivateRaiseToTheSixthPowerActivation() --> Derivative of raise to the 6th power activation function.
* 10) getDerivateFirstOrderDegreeExponentialActivation() --> Derivative of 1st order degree exponential activation function.
* 11) getDerivateSecondOrderDegreeExponentialActivation() --> Derivative of 2nd order degree exponential activation function.
* For all these functions that apply a derivate, the following will
* explain how to use their argument variables and what considerations
* must have be taken into account:
*
* @param double *u - This argument will contain the pointer to a
*					 memory allocated input matrix, in which the
*					 output of the body of a neuron should be
*					 stored. THIS VARIABLE SHOULD BE ALLOCATED AND
*					 INITIALIZED BEFORE CALLING THIS FUNCTION
*					 WITH A SIZE OF "n" TIMES "p=1" 'DOUBLE'
*					 MEMORY SPACES.
*
* @param double *A_u - This argument will contain the pointer to a
*					 memory allocated input matrix in which a
*					 previously requested activation function was
*					 applied and stored in it with respect to the
*					 pointer argument vriable "u". "A_u" SHOULD BE
*					 ALLOCATED BEFORE CALLING THIS FUNCTION WITH A
*					 SIZE OF "n" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param double *dA_u - This argument will contain the pointer to a
*					 memory allocated output matrix in which any of
*					 these functions will store the result of
*					 applying the requested derivative of a
*					 particular activation function with respect to
*					 the pointer argument variable "A_u". "dA_u"
*					 SHOULD BE ALLOCATED BEFORE CALLING THIS
*					 FUNCTION WITH A SIZE OF "n" TIMES "p=1"
*					 'DOUBLE' MEMORY SPACES.
*
* @param struct singleNeuronDnnStruct *neuron - This argument will
*					 contain the pointer to a struct variable that
*					 should contain all the information required in
*					 order to be able to create and make an
*					 artificial neuron. Its accessible inner
*					 elements are described in the list showed in
*					 the commented documentation of the function
*					 "getSingleNeuronDNN()".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"dA_u".
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
static void getDerivateLogisticActivation(double *u, double *A_u, double *dA_u, struct singleNeuronDnnStruct *neuron) {
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
* The "getNeuronAdjustedCoefficientOfDetermination()" static function
* is used to apply a regression evaluation metric known as the
* adjusted coefficient of determination. Such method will be applied
* with respect to the argument pointer variables "realOutputMatrix"
* and "predictedOutputMatrix". Then, its result will be stored in the
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
*									INITIALIZED BEFORE CALLING THIS
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
*										 BE ALLOCATED AND INITIALIZED
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
* CREATION DATE: NOVEMBER 29, 2021
* LAST UPDATE: DECEMBER 05, 2021
*/
static void getNeuronAdjustedCoefficientOfDetermination(double *realOutputMatrix, double *predictedOutputMatrix, int n, int m, int degreesOfFreedom, double *adjustedRsquared) {
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


/**
* The "getNeuronAccuracy()" static function is used to calculate and
* obtain the classification evaluation metric known as the accuracy.
* Such method will be applied with respect to the argument pointer
* variables "realOutputMatrix" and "predictedOutputMatrix". Then, its
* result will be stored in the argument pointer variable "accuracy".
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
*									AND INITIALIZED BEFORE CALLING
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
*										 INITIALIZED BEFORE CALLING
*										 THIS FUNCTION WITH A SIZE OF
*										 "n" TIMES "p" 'DOUBLE'
*										 MEMORY SPACES.
*
* @param int n - This argument will represent the total number of 
*				 samples (rows) that the input matrix has, with which 
*				 the output data was obtained.
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
*							FUNCTION WITH A SIZE OF "1" 'DOUBLE' MEMORY
*							SPACES where the result will be stored.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "accuracy".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 29, 2021
* LAST UPDATE: DECEMBER 05, 2021
*/
static void getNeuronAccuracy(double *realOutputMatrix, double *predictedOutputMatrix, int n, double *accuracy) {
	// In order to calculate the accuracy, we calculate the true positives and true negatives between the argument pointer variables "realOutputMatrix" and "predictedOutputMatrix".
	double tp = 0; // Variable used to store the true positives.
	double tn = 0; // Variable used to store the true negatives.
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
* The "predictSingleNeuronDNN()" function is used to make the
* predictions of the requested input values (X) by applying the
* simple linear equation system with the specified coefficient values
* (b). The predicted values will be stored in the argument pointer
* variable "Y_hat".
* 
* @param struct singleNeuronDnnStruct *neuron - This argument will
*					 contain the pointer to a struct variable that
*					 should contain all the information required in
*					 order to be able to create and make an
*					 artificial neuron. Its accessible inner
*					 elements are described in the list showed in
*					 the commented documentation of the function
*					 "getSingleNeuronDNN()".
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
* CREATION DATE: NOVEMBER 29, 2021
* LAST UPDATE: JANUARY 09, 2022
*/
void predictSingleNeuronDNN(struct singleNeuronDnnStruct *neuron, double *Y_hat) {
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
	
	// --------------- BEGINNING OF PREDICTION PROCESS --------------- //
	// We calculate the currently predicted output data made by the body of the neuron and store it in "f_x_tilde". However, for performance purposes, we will temporarily store the values of "f_x_tilde" in "Y_hat".
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow=0; currentRow<neuron->n; currentRow++) {
		Y_hat[currentRow] = neuron->w_best[0];
		currentRowTimesM = currentRow*neuron->m;
		for (int currentColumn=1; currentColumn<(neuron->m+1); currentColumn++) {
			Y_hat[currentRow] = Y_hat[currentRow] + neuron->w_best[currentColumn]*neuron->X[currentColumn-1 + currentRowTimesM];
		}
	}
	
	// We calculate, in its continous (regression) form, the currently predicted output data made by the neuron and store it in "Y_hat" by applying the desired activation function to "f_x_tilde".
	// NOTE: Remember that "Y_hat" = A(u) = "A_u".
	// NOTE: "activationFunctions" is a pointer to each of the individual activation functions that were developed as static void functions.
    static void (*activationFunctions[])(double *, double *, struct singleNeuronDnnStruct *) = {getReluActivation, getTanhActivation, getLogisticActivation, getRaiseToTheFirstPowerActivation, getRaiseToTheSecondPowerActivation, getRaiseToTheThirdPowerActivation, getRaiseToTheFourthPowerActivation, getRaiseToTheFifthPowerActivation, getRaiseToTheSixthPowerActivation, getFirstOrderDegreeExponentialActivation, getSecondOrderDegreeExponentialActivation};
	(*activationFunctions[neuron->activationFunctionToBeUsed])(Y_hat, Y_hat, neuron); // We calculate A(u) and store it in the pointer variable "Y_hat".
	
	// Determine if the given model of a single neuron in Deep Neural Network is meant for a classification or for a regression problem to then make the predictions accordingly.
	if (neuron->isClassification == 1) {
		// We apply the threshold define by the implementer in order to obtain a classification output and store it in "Y_hat".
		for (int currentRow=0; currentRow<(neuron->n); currentRow++) {
			if (Y_hat[currentRow] > neuron->threshold) {
				Y_hat[currentRow] = neuron->desiredValueForGroup1; // Group 1 has been predicted.
			} else {
				Y_hat[currentRow] = neuron->desiredValueForGroup2; // Group 2 has been predicted.
			}
		}
	}
	
	return;
}

