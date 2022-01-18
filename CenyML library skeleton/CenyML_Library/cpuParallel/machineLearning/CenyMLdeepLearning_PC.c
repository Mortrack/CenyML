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
#include "CenyMLdeepLearning_PC.h"
// IMPORTANT NOTE: This library uses the math.h library and therefore,
//				   remember to use the "-lm" flag when compiling it.
//				   In addition, remember to use the "-lpthread" when
//				   compiling it because this library uses the pthread.h
//				   library.

// NOTE: "activationFunctions" is a pointer to each of the individual activation functions that were developed as static void functions.
static void (*activationFunctions[])(void *) = {getReluActivation_parallelCPU, getTanhActivation_parallelCPU, getLogisticActivation_parallelCPU, getRaiseToTheFirstPowerActivation_parallelCPU, getRaiseToTheSecondPowerActivation_parallelCPU, getRaiseToTheThirdPowerActivation_parallelCPU, getRaiseToTheFourthPowerActivation_parallelCPU, getRaiseToTheFifthPowerActivation_parallelCPU, getRaiseToTheSixthPowerActivation_parallelCPU, getFirstOrderDegreeExponentialActivation_parallelCPU, getSecondOrderDegreeExponentialActivation_parallelCPU};
// NOTE: "derivateOfActivationFunctions" is a pointer to each of the individual derivatives of the activation functions that were developed as static void functions.
static void (*derivateOfActivationFunctions[])(void *) = {getDerivateReluActivation_parallelCPU, getDerivateTanhActivation_parallelCPU, getDerivateLogisticActivation_parallelCPU, getDerivateRaiseToTheFirstPowerActivation_parallelCPU, getDerivateRaiseToTheSecondPowerActivation_parallelCPU, getDerivateRaiseToTheThirdPowerActivation_parallelCPU, getDerivateRaiseToTheFourthPowerActivation_parallelCPU, getDerivateRaiseToTheFifthPowerActivation_parallelCPU, getDerivateRaiseToTheSixthPowerActivation_parallelCPU, getDerivateFirstOrderDegreeExponentialActivation_parallelCPU, getDerivateSecondOrderDegreeExponentialActivation_parallelCPU};
	
struct cpuParallelData {
	struct singleNeuronDnnStruct_parallelCPU neuronData; // structure variable that will contain all the input data that was previously given to the neuron.
	int threadId; // Variable used to store the unique identifier that is given to each CPU thread to be created.
	int threadStart; // Variable used to store the starting working points for each of the CPU threads to be created.
	int threadStop; // Variable used to store the working points at which each of the CPU threads to be created will stop.
	int mPlusOne; //This variable is used to store a repetitive matheamtical operation, for performance purposes.
	double *TransposeOf_X_tilde; // Pointer variable that will contain the transpose of the input data from which the desired machine learning method will be calcualted. THIS VARIABLE SHOULD BE ALLOCATED AND INITIALIZED WITH A SIZE OF "m+1" TIMES "n" 'DOUBLE' MEMORY SPACES.
	double *f_x_tilde; // Allocate the memory required for the variable "f_x_tilde", which will contain the currently predicted output data made by the body of the neuron. THIS VARIABLE SHOULD BE ALLOCATED AND INITIALIZED WITH A SIZE OF "n" TIMES "1" 'DOUBLE' MEMORY SPACES.
	double *A_u; // Allocate the memory required for the variable "A_u", which will contain the currently predicted output data made by the neuron. THIS VARIABLE SHOULD BE ALLOCATED AND INITIALIZED WITH A SIZE OF "n" TIMES "1" 'DOUBLE' MEMORY SPACES.
	double *dA_u; // Allocate the memory required for the variable "dA_u", which will contain the derivative of A(u). THIS VARIABLE SHOULD BE ALLOCATED AND INITIALIZED WITH A SIZE OF "n" TIMES "1" 'DOUBLE' MEMORY SPACES.
	double accuracyTerm1; // Declare the variable "accuracyTerm1", which will contain the term that each of the threads will store and that will be used to calculate the current accuracy (this particular term will be SSE or tp).
	double accuracyTerm2; // Declare the variable "accuracyTerm2", which will contain the term that each of the threads will store and that will be used to calculate the current accuracy (this particular term will be SST or tn).
	double *w_old; // Allocate the memory required for the variable "w_old", which will contain the previous weight values that were obtained with respect to the current ones. THIS VARIABLE SHOULD BE ALLOCATED AND INITIALIZED WITH A SIZE OF "m+1" TIMES "1" 'DOUBLE' MEMORY SPACES.
	double *errorTerm; // Allocate the memory required for the variable "errorTerm", which will contain the current error term value to be taken into consideration for the update of the weight values. THIS VARIABLE SHOULD BE ALLOCATED AND INITIALIZED WITH A SIZE OF "n" TIMES "1" 'DOUBLE' MEMORY SPACES.
	double *errorTerm_dot_Xtilde; // Allocate the memory required for the variable "errorTerm_dot_Xtilde", which will contain the resulting dot product between the error term and the transpose of "X_tilde". THIS VARIABLE SHOULD BE ALLOCATED AND INITIALIZED WITH A SIZE OF "m+1" TIMES "number of cpu threads to be used" 'DOUBLE' MEMORY SPACES.
};


/**
* The "getSingleNeuronDNN_parallelCPU()" function is used to apply
* the machine learning algorithm called single neuron in Deep
* Neural Network as formulated in the master thesis of Cesar
* Miranda Meza called "Machine learning to support applications
* with embedded systems and parallel computing", but in its CPU
* parallel version. Within this process, the best fitting equation
* with the form of "y_hat = b_0 + w_1*x_1 + w_2*x_2 + ... + w_m*x_m"
* will be identified with respect to the sampled data given through
* the argument pointer variables "X" and "Y". As a result, the
* identified coefficient values will be stored in the argument
* pointer variable "neuron->w_new". With respect to the struct
* pointer variable "neuron", it should contain all the information
* required in order to be able to create and make an artificial
* neuron. Its accessible inner elements will be described in the
* following list:
*
*
* @param int cpuThreads - This argument will represent the desired
*						  number of threads with which the
*						  implementer wants this algorithm to
*						  parallelize by using the CPU of the
*						  computational system in which this
*						  algorithm is being executed.
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
*								 expected from it to interpret the given
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
*										and this function will end. Note that the
*										evaluation metric to be used will be the
*										adjusted R squared regardless if the data
*										is for classification or not.
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
* CREATION DATE: JANUARY 17, 2022
* LAST UPDATE: N/A
*/
void getSingleNeuronDNN_parallelCPU(struct singleNeuronDnnStruct_parallelCPU *neuron) {
	// If the CPU threads are less than value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->cpuThreads < 1) {
		printf("\nERROR: The requested CPU threads must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the CPU threads are greater than the number of samples, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->cpuThreads) > (neuron->n)) {
		printf("\nERROR: The requested CPU threads must be equal or less than the number of machine learning samples for this particular algorithm.\n");
		exit(1);
	}
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
	// We initialize the data that we will give to each of the CPU threads to be created.
	struct cpuParallelData threadData[neuron->cpuThreads]; // We create a struct variable used to store all the data that will be required by the CPU threads to be created.
	int mPlusOne = neuron->m + 1; // This value is repetitively used and strategically stored here for performance purposes.
	double *TransposeOf_X_tilde = (double *) malloc(mPlusOne*neuron->n*sizeof(double)); // We allocate the memory required for the local pointer variable that will contain the input data from which the desired machine learning method will be calcualted.
	for (int currentThread; currentThread<(neuron->cpuThreads); currentThread++) {
		threadData[currentThread].neuronData = neuron[0]; // We give the neuron's input data to each CPU thread.
		threadData[currentThread].threadStart = currentThread * neuron->n / neuron->cpuThreads; // We assign the starting working point of each thread.
		threadData[currentThread].threadStop = (currentThread+1) * neuron->n / neuron->cpuThreads; // We assign the working point at which each thread will stop.
		threadData[currentThread].mPlusOne = mPlusOne; // We give the directory address of the variable that contains the value that has to be in the "mPlusOne" pointer variable.
		threadData[currentThread].TransposeOf_X_tilde = TransposeOf_X_tilde; // We pass the pointer of the variable that will store the transpose of the input data from which the desired machine learning method will be calcualted.
	}
	
	// We obtain and store the transpose of "X_tilde".
	pthread_t threadId[neuron->cpuThreads]; // pthread_t object definition with an integer array structure to be used as an ID for each created thread.
	for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We create the specified threads and assign them the function they will work with.
		pthread_create(&threadId[currentThread], NULL, getTransposeOfInputData_parallelCPU, &threadData[currentThread]);
	}
	
	
	// -------------------- WEIGHT INITIALIZATION -------------------- //
	// Store the initial weight values into "neuron->w_new" and into "neuron->w_best" sequentially.
	if (neuron->isInitial_w == 0) {
		// In order to initialize "neuron->w_new" with random values between "-1" and "+1"., intialize random number generator.
	    time_t t;
		srand((unsigned) time(&t));
	    
	    // Initialize "neuron->w_new" with random values between -1 to +1 with three decimals at the most. Give the save values to "neuron->w_best".
	    double currentRandomNumber;
	    for (int current_w=0 ; current_w<mPlusOne; current_w++) {
	        currentRandomNumber = ((float) (rand() % 1000))/500 - 1;
	        neuron->w_first[current_w] = currentRandomNumber;
	        neuron->w_new[current_w] = currentRandomNumber;
	        neuron->w_best[current_w] = currentRandomNumber;
	    }
	} else if (neuron->isInitial_w == 1) {
		// Pass the values of "neuron->w_first" to "neuron->w_new" and "neuron->w_best".
		for (int current_w=0 ; current_w<mPlusOne; current_w++) {
	        neuron->w_new[current_w] = neuron->w_first[current_w];
	        neuron->w_best[current_w] = neuron->w_first[current_w];
	    }
	}
	
	// We allocate all the memory that will be required for the training process of the neuron.
	double *f_x_tilde = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "f_x_tilde", which will contain the currently predicted output data made by the body of the neuron.
    double *A_u = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "A_u", which will contain the currently predicted output data made by the neuron.
    double *dA_u = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "dA_u", which will contain the derivative of A(u).
	double totalSumOfAccuracyTerm1 = 0; // This variable is used to sum all the contributions of each CPU thread that were made to get the accuracy term 1.
	double totalSumOfAccuracyTerm2 = 0; // This variable is used to sum all the contributions of each CPU thread that were made to get the accuracy term 2.
	double currentAccuracy = 0; // Declare the variable "currentAccuracy", which will contain the current accuracy of the neuron.
    double *w_old = (double *) malloc(mPlusOne*neuron->p*sizeof(double)); // Allocate the memory required for the variable "w_old", which will contain the previous weight values that were obtained with respect to the current ones.
	double *errorTerm = (double *) malloc(neuron->n*neuron->p*sizeof(double)); // Allocate the memory required for the variable "errorTerm", which will contain the current error term value to be taken into consideration for the update of the weight values.
	double *errorTerm_dot_Xtilde = (double *) malloc(mPlusOne*neuron->cpuThreads*sizeof(double)); // Allocate the memory required for the variable "errorTerm_dot_Xtilde", which will contain all the individual contributions of each CPU thread that calculated the resulting dot product between the error term and the transpose of "X_tilde".
	double totalErrorTerm_dot_Xtilde = 0; // This variable is used to store the sum of each of the error terms that were obtained by each of the CPU threads.
	int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesCpuThreads; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentThread; currentThread<(neuron->cpuThreads); currentThread++) { // We pass the pointers of all the variables that will be used by the CPU threads to be created.
		threadData[currentThread].threadId = currentThread; // We give the identifier to the current CPU thread.
		threadData[currentThread].f_x_tilde = f_x_tilde; // We pass the pointer of the variable that will store the currently predicted output data made by the body of the neuron.
		threadData[currentThread].A_u = A_u; // We pass the pointer of the variable that will store the currently predicted output data made by the neuron.
		threadData[currentThread].dA_u = dA_u; // We pass the pointer of the variable that will store the derivative of A(u).
		threadData[currentThread].w_old = w_old; // We pass the pointer of the variable that will store the previous weight values that were obtained with respect to the current ones.
		threadData[currentThread].errorTerm = errorTerm; // We pass the pointer of the variable that will store the current error term value to be taken into consideration for the update of the weight values.
		threadData[currentThread].errorTerm_dot_Xtilde = errorTerm_dot_Xtilde; // We pass the pointer of the variable that will store all the individual contributions of each CPU thread that calculated the resulting dot product between the error term and the transpose of "X_tilde".
	}
	for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We force the program to wait until all threads have finished their assigned task.
		pthread_join(threadId[currentThread], NULL);
	}
	
	// ------------------------------------- //
	// ----- REGRESSION MODEL SELECTED ----- //
	// ------------------------------------- //


	// ----------- EVALUATION OF THE INITIAL WEIGHT VALUES ----------- //
	// We calculate "f_x_tilde", "A(u)", "dA(u)" and "the part 1 of the accuracy terms".
	for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We create the specified threads and assign them the function they will work with.
		pthread_create(&threadId[currentThread], NULL, getFxTilde_Au_dAu_and_accuracyTermsPart1_parallelCPU, &threadData[currentThread]);
	}
	for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We force the program to wait until all threads have finished their assigned task.
		pthread_join(threadId[currentThread], NULL);
	}

	// We calculate "the part 1 of the accuracy terms".
	totalSumOfAccuracyTerm1 = 0; // We reset the value of the accuracy term 1, in which we will store the value of SSE.
	for (int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) {
		totalSumOfAccuracyTerm1 += threadData[currentThread].accuracyTerm1; // We sum all the fragments of the SSE that was calculated by the previous parallelization process.
	}
	for (int currentThread=1; currentThread<(neuron->cpuThreads); currentThread++) {
		threadData[0].accuracyTerm2 += threadData[currentThread].accuracyTerm2; // We sum all the fragments of the "real output matrix sum" that was calculated by the previous parallelization process.
	}
	threadData[0].accuracyTerm2 = threadData[0].accuracyTerm2 / neuron->n; // We calculate the mean of the "real output matrix".
	for (int currentThread=1; currentThread<(neuron->cpuThreads); currentThread++) {
		threadData[currentThread].accuracyTerm2 = threadData[0].accuracyTerm2; // We pass the total mean of the "real output matrix" to all the data of all the CPU threads.
	}

	// We calculate "the part 2 of the accuracy terms".
	totalSumOfAccuracyTerm2 = 0; // We reset the value of the accuracy term 2, in which we will store the value of SST.
	for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We create the specified threads and assign them the function they will work with.
		pthread_create(&threadId[currentThread], NULL, getNeuronAdjustedCoefficientOfDetermination_parallelCPUvoidPart2, &threadData[currentThread]);
	}
	for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We force the program to wait until all threads have finished their assigned task.
		pthread_join(threadId[currentThread], NULL);
	}
	for (int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) {
		totalSumOfAccuracyTerm2 += threadData[currentThread].accuracyTerm1; // We sum all the fragments of the SST that was calculated by the previous parallelization process.
	}

	// Finally, we calculate the adjusted coefficient of determination and store its results in the pointer variable "adjustedRsquared".
	currentAccuracy = 1 - ( (totalSumOfAccuracyTerm1/((neuron->n)-(neuron->m)-1))/(totalSumOfAccuracyTerm2/((neuron->n)-1)) );

	// We pass the current accuracy to the best accuracy record because this is the evaluation of the very first weight values.
	neuron->bestAccuracy = currentAccuracy;

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
		for (int currentCoefficient=0; currentCoefficient<mPlusOne; currentCoefficient++) {
			w_old[currentCoefficient] = neuron->w_new[currentCoefficient];
		}
		
		// Calculate the error term obtainable with the current weight values so that we can later update the current weight values ("w_old") in order to obtain the new ones ("neuron->w_new").
		for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We create the specified threads and assign them the function they will work with.
			pthread_create(&threadId[currentThread], NULL, getErrorAndUpdateWeightValues_parallelCPU, &threadData[currentThread]);
		}
		for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We force the program to wait until all threads have finished their assigned task.
			pthread_join(threadId[currentThread], NULL);
		}
		
		// We update the current weight values ("w_old") in order to obtain the new ones ("neuron->w_new") by summing all the individual contributions made by the previous parallelization process.
		for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
			totalErrorTerm_dot_Xtilde = 0;
			currentRowTimesCpuThreads = currentRow * neuron->cpuThreads;
			for (int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) {
				totalErrorTerm_dot_Xtilde += errorTerm_dot_Xtilde[currentThread + currentRowTimesCpuThreads];
			}
			neuron->w_new[currentRow] = w_old[currentRow] + neuron->learningRate * totalErrorTerm_dot_Xtilde;
		}
		
		// We recalculate "f_x_tilde", "A(u)", "dA(u)" and "the part 1 of the accuracy terms".
		for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We create the specified threads and assign them the function they will work with.
			pthread_create(&threadId[currentThread], NULL, getFxTilde_Au_dAu_and_accuracyTermsPart1_parallelCPU, &threadData[currentThread]);
		}
		for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We force the program to wait until all threads have finished their assigned task.
			pthread_join(threadId[currentThread], NULL);
		}
		
		// We recalculate "the part 1 of the accuracy terms".
		totalSumOfAccuracyTerm1 = 0; // We reset the value of the accuracy term 1, in which we will store the value of SSE.
		for (int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) {
			totalSumOfAccuracyTerm1 += threadData[currentThread].accuracyTerm1; // We sum all the fragments of the SSE that was calculated by the previous parallelization process.
		}
		for (int currentThread=1; currentThread<(neuron->cpuThreads); currentThread++) {
			threadData[0].accuracyTerm2 += threadData[currentThread].accuracyTerm2; // We sum all the fragments of the "real output matrix sum" that was calculated by the previous parallelization process.
		}
		threadData[0].accuracyTerm2 = threadData[0].accuracyTerm2 / neuron->n; // We calculate the mean of the "real output matrix".
		for (int currentThread=1; currentThread<(neuron->cpuThreads); currentThread++) {
			threadData[currentThread].accuracyTerm2 = threadData[0].accuracyTerm2; // We pass the total mean of the "real output matrix" to all the data of all the CPU threads.
		}
		
		// We recalculate "the part 2 of the accuracy terms".
		totalSumOfAccuracyTerm2 = 0; // We reset the value of the accuracy term 2, in which we will store the value of SST.
		for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We create the specified threads and assign them the function they will work with.
			pthread_create(&threadId[currentThread], NULL, getNeuronAdjustedCoefficientOfDetermination_parallelCPUvoidPart2, &threadData[currentThread]);
		}
		for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We force the program to wait until all threads have finished their assigned task.
			pthread_join(threadId[currentThread], NULL);
		}
		for (int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) {
			totalSumOfAccuracyTerm2 += threadData[currentThread].accuracyTerm1; // We sum all the fragments of the SST that was calculated by the previous parallelization process.
		}
		
		// Finally, we recalculate the adjusted coefficient of determination and store its results in the pointer variable "adjustedRsquared".
		currentAccuracy = 1 - ( (totalSumOfAccuracyTerm1/((neuron->n)-(neuron->m)-1))/(totalSumOfAccuracyTerm2/((neuron->n)-1)) );
		
		// We compare the accuracy of the currently obtained weight values with respect to the latest best one recorded. If the current one is better than the recorded one, then store the current one in its place and do the same for the best recorded weight values.
		if ((currentAccuracy) > (neuron->bestAccuracy)) {
			neuron->bestAccuracy = currentAccuracy; // Pass the value of the current accuracy into "neuron->bestAccuracy".
			for (int current_w=0 ; current_w<mPlusOne; current_w++) { // Pass the values of "neuron->w_new" to "neuron->w_best".
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
	
	// Determine whether it was requested that the neuron reports its learning progress or not.
	if (neuron->isReportLearningProgress == 1) { // If the implementer requested the neuron to report its progress, apply the following code.
		// Make the neuron report its last progress made.
		printf("\nEpoch %d --> single neuron in DNN has achieved an adjusted R squared of %f\n", neuron->maxEpochs, currentAccuracy);
	}
	
	// Before terminating this function, we free the Heap memory used for the allocated variables since they will no longer be used.
	free(TransposeOf_X_tilde);
	free(f_x_tilde);
	free(A_u);
	free(dA_u);
	free(w_old);
	free(errorTerm);
	free(errorTerm_dot_Xtilde);
	
	printf("\nThe best adjusted R squared (%f) achieved by the neuron did not surpased the defined goal but its training process has been successfully concluded.\n", neuron->bestAccuracy);
	
	return;
}


/**
* The "getTransposeOfInputData_parallelCPU()" static function is used
* to calculate and store the transpose of the input matrix that will
* be used to train a single artificial neuron, but in its CPU parallel
* version.
* 
* To learn more about the argument cpuParallelData structure variable
* that is used in this function, read the code comments that are
* located in first lines written in this file.
*
*
* NOTE: RESULTS ARE STORED IN "threadData->TransposeOf_X_tilde".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 14, 2022
* LAST UPDATE: N/A
*/
static void *getTransposeOfInputData_parallelCPU(void *threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate and store the corresponding part of the transpose of "X_tilde" that was assigned to the current CPU thread.
	int currentRowTimesMplusOne; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRow2; // This variable is used in the for-loop for the matrix transpose that will be made.
	int currentColumn2 = threadData->threadStart; // This variable is used in the for-loop for the matrix transpose that will be made.
	for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		currentRow2 = 0; // We reset the counters used in the following for-loop.
		currentRowTimesMplusOne = currentRow*(threadData->mPlusOne);
		currentRowTimesM = currentRow*(threadData->neuronData.m);
		threadData->TransposeOf_X_tilde[currentColumn2] = 1;
		for (int currentColumn=1; currentColumn<(threadData->mPlusOne); currentColumn++) {
			currentRow2++;
			threadData->TransposeOf_X_tilde[currentColumn2 + currentRow2*(threadData->neuronData.n)] = threadData->neuronData.X[currentColumn-1 + currentRowTimesM];
		}
		currentColumn2++;
	}
	
	return NULL;
}
/**
* The "getFxTilde_Au_dAu_and_accuracyTermsPart1_parallelCPU()" static
* function is used to calculate "\tilde{f}_{x}", A(u), dA(u) and the
* first part of the accuracy terms through the application of CPU
* parallelism.
* 
* To learn more about the argument cpuParallelData structure variable
* that is used in this function, read the code comments that are
* located in first lines written in this file.
*
*
* NOTE: RESULTS ARE STORED IN "threadData->f_x_tilde",
* 		"threadData->A_u", "threadData->dA_u",
* 		"threadData->accuracyTerm1" and "threadData->accuracyTerm1".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 17, 2022
* LAST UPDATE: N/A
*/
static void *getFxTilde_Au_dAu_and_accuracyTermsPart1_parallelCPU(void *threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the corresponding currently predicted output data made by the body of the neuron and store it in "threadData->f_x_tilde" and that was assigned to the current CPU thread.
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->f_x_tilde[currentRow] = threadData->neuronData.w_new[0];
		currentRowTimesM = currentRow * threadData->neuronData.m;
		for (int currentColumn=1; currentColumn<(threadData->mPlusOne); currentColumn++) {
			threadData->f_x_tilde[currentRow] = threadData->f_x_tilde[currentRow] + threadData->neuronData.w_new[currentColumn] * threadData->neuronData.X[currentColumn-1 + currentRowTimesM];
		}
	}
	
	// We calculate the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
	(*activationFunctions[threadData->neuronData.activationFunctionToBeUsed])(threadVariable); // We calculate A(u) and store it in the pointer variable "A_u".
	
	// We calculate the derivative of A(u).
    // NOTE: Remember that "Y_hat" = A(u) = "A_u".
	(*derivateOfActivationFunctions[threadData->neuronData.activationFunctionToBeUsed])(threadVariable); // We calculate the derivative of A(u) and store it in the pointer variable "dA_u".
	
	// We calculate the corresponding evaluation metric with respect to the actual data of the system under study "neuron->Y" and the currently predicted output made by the neuron "A_u".
	getNeuronAdjustedCoefficientOfDetermination_parallelCPUvoidPart1(threadVariable); // We calculate the part 1 of the calculation of the current adjusted coefficient of determination of the neuron.
	
	return NULL;
}
/**
* The "getErrorAndUpdateWeightValues_parallelCPU()" static function is
* used to calculate the error term obtainable with the current weight
* values by applying CPU parallelism
* 
* To learn more about the argument cpuParallelData structure variable
* that is used in this function, read the code comments that are
* located in first lines written in this file.
*
*
* NOTE: RESULTS ARE STORED IN "threadData->errorTerm_dot_Xtilde".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 15, 2022
* LAST UPDATE: N/A
*/
static void *getErrorAndUpdateWeightValues_parallelCPU(void *threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// Calculate the error term obtainable with the current weight values.
	for (int currentSample=(threadData->threadStart); currentSample<(threadData->threadStop); currentSample++) {
		threadData->errorTerm[currentSample] = (threadData->neuronData.Y[currentSample] - threadData->A_u[currentSample]) * threadData->dA_u[currentSample];
	}
	
	// We update the current weight values ("w_old") in order to obtain the new ones ("neuron->w_new").
	int currentThreadPlusCurrentRowTimesCpuThreads; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	int currentRowTimesN; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow=0; currentRow<(threadData->mPlusOne); currentRow++) {
		currentThreadPlusCurrentRowTimesCpuThreads = threadData->threadId + currentRow * threadData->neuronData.cpuThreads;
		threadData->errorTerm_dot_Xtilde[currentThreadPlusCurrentRowTimesCpuThreads] = 0;
		currentRowTimesN = currentRow*threadData->neuronData.n;
		for (int currentSample=(threadData->threadStart); currentSample<(threadData->threadStop); currentSample++) {
			// We first multiply all the samples of the "errorTerm" with all the samples of the transpose of "X_tilde".
			threadData->errorTerm_dot_Xtilde[currentThreadPlusCurrentRowTimesCpuThreads] += threadData->errorTerm[currentSample] * threadData->TransposeOf_X_tilde[currentSample + currentRowTimesN];
		}
	}

	return NULL;
}


/**
* The following static functions have the purpose of applying the
* requested activation function and/or derivative of such activation
* function by using CPU parallelism. In this regard, the list of all
* the static functions that will apply an activation function, are
* the following:
*
* 1) getReluActivation_parallelCPU() --> Applies the ReLU activation function.
* 2) getTanhActivation_parallelCPU() --> Applies the tanh activation function.
* 3) getLogisticActivation_parallelCPU() --> Applies the Logistic activation function.
* 4) getRaiseToTheFirstPowerActivation_parallelCPU() --> Applies the raise to the 1st power activation function.
* 5) getRaiseToTheSecondPowerActivation_parallelCPU() --> Applies the raise to the 2nd power activation function.
* 6) getRaiseToTheThirdPowerActivation_parallelCPU() --> Applies the raise to the 3rd power activation function.
* 7) getRaiseToTheFourthPowerActivation_parallelCPU() --> Applies the raise to the 4th power activation function.
* 8) getRaiseToTheFifthPowerActivation_parallelCPU() --> Applies the raise to the 5th power activation function.
* 9) getRaiseToTheSixthPowerActivation_parallelCPU() --> Applies the raise to the 6th power activation function.
* 10) getFirstOrderDegreeExponentialActivation_parallelCPU() --> Applies the 1st order degree exponential activation function.
* 11) getSecondOrderDegreeExponentialActivation_parallelCPU() --> Applies the 2nd order degree exponential activation function.
* For all these functions that apply a derivate, the following will
* explain how to use their argument variables and what considerations
* must have be taken into account:
*
* To learn more about the argument cpuParallelData structure variable
* that is used in this function, read the code comments that are
* located in first lines written in this file.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"threadData->A_u".
* 
* @return void
*
* ------------------------------------------------------------------
* ------------------------------------------------------------------
*
* On the other hand, the list of all the static functions that will
* apply the derivative of such activation functions, are the following:
*
* 1) getDerivateReluActivation_parallelCPU() --> Derivative of ReLU activation function.
* 2) getDerivateTanhActivation_parallelCPU() --> Derivative of tanh activation function.
* 3) getDerivateLogisticActivation_parallelCPU() --> Derivative of Logistic activation function.
* 4) getDerivateRaiseToTheFirstPowerActivation_parallelCPU() --> Derivative of raise to the 1st power activation function.
* 5) getDerivateRaiseToTheSecondPowerActivation_parallelCPU() --> Derivative of raise to the 2nd power activation function.
* 6) getDerivateRaiseToTheThirdPowerActivation_parallelCPU() --> Derivative of raise to the 3rd power activation function.
* 7) getDerivateRaiseToTheFourthPowerActivation_parallelCPU() --> Derivative of raise to the 4th power activation function.
* 8) getDerivateRaiseToTheFifthPowerActivation_parallelCPU() --> Derivative of raise to the 5th power activation function.
* 9) getDerivateRaiseToTheSixthPowerActivation_parallelCPU() --> Derivative of raise to the 6th power activation function.
* 10) getDerivateFirstOrderDegreeExponentialActivation_parallelCPU() --> Derivative of 1st order degree exponential activation function.
* 11) getDerivateSecondOrderDegreeExponentialActivation_parallelCPU() --> Derivative of 2nd order degree exponential activation function.
* For all these functions that apply a derivate, the following will
* explain how to use their argument variables and what considerations
* must have be taken into account:
*
* To learn more about the argument cpuParallelData structure variable
* that is used in this function, read the code comments that are
* located in first lines written in this file.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*		"threadData->dA_u".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 15, 2022
* LAST UPDATE: N/A
*/
// ------------------ RELU ACTIVATION FUNCTION ------------------- //
static void getReluActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		if (threadData->f_x_tilde[currentRow] > 0) {
	        threadData->A_u[currentRow] = threadData->f_x_tilde[currentRow];
	    } else {
	        threadData->A_u[currentRow] = 0;
	    }
	}
}
static void getDerivateReluActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		if (threadData->f_x_tilde[currentRow] > 0) {
	        threadData->dA_u[currentRow] = 1;
	    } else {
	        threadData->dA_u[currentRow] = 0;
	    }
	}
}

// ------------------ TANH ACTIVATION FUNCTION ------------------- //
static void getTanhActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->A_u[currentRow] = (exp(threadData->f_x_tilde[currentRow]) - exp(-threadData->f_x_tilde[currentRow])) / (exp(threadData->f_x_tilde[currentRow]) + exp(-threadData->f_x_tilde[currentRow]));
	}
}
static void getDerivateTanhActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->dA_u[currentRow] = 1 - threadData->A_u[currentRow] * threadData->A_u[currentRow];
	}
}

// ---------- RAISE TO THE LOGISTIC POWER ACTIVATION FUNCTION --------- //
static void getLogisticActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->A_u[currentRow] = 1 / (1 + exp(-threadData->f_x_tilde[currentRow]));
	}
}
static void getDerivateLogisticActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->dA_u[currentRow] = threadData->A_u[currentRow] * (1 - threadData->A_u[currentRow]);
	}
}

// ---------- RAISE TO THE 1ST POWER ACTIVATION FUNCTION --------- //
static void getRaiseToTheFirstPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->A_u[currentRow] = threadData->f_x_tilde[currentRow];
	}
}
static void getDerivateRaiseToTheFirstPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->dA_u[currentRow] = 1;
	}
}

// --------- RAISE TO THE 2ND POWER ACTIVATION FUNCTION ---------- //
static void getRaiseToTheSecondPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->A_u[currentRow] = threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow];
	}
}
static void getDerivateRaiseToTheSecondPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->dA_u[currentRow] = 2*threadData->f_x_tilde[currentRow];
	}
}

// --------- RAISE TO THE 3RD POWER ACTIVATION FUNCTION ---------- //
static void getRaiseToTheThirdPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->A_u[currentRow] = threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow];
	}
}
static void getDerivateRaiseToTheThirdPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->dA_u[currentRow] = 3 * threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow];
	}
}

// --------- RAISE TO THE 4TH POWER ACTIVATION FUNCTION ---------- //
static void getRaiseToTheFourthPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    double squareThisValue; // Variable used to store the value that wants to be squared.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
    	squareThisValue = threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow];
		threadData->A_u[currentRow] = squareThisValue * squareThisValue;
	}
}
static void getDerivateRaiseToTheFourthPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->dA_u[currentRow] = 4 * threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow];
	}
}

// --------- RAISE TO THE 5TH POWER ACTIVATION FUNCTION ---------- //
static void getRaiseToTheFifthPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    double squareThisValue; // Variable used to store the value that wants to be squared.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
    	squareThisValue = threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow];
		threadData->A_u[currentRow] = squareThisValue * squareThisValue * threadData->f_x_tilde[currentRow];
	}
}
static void getDerivateRaiseToTheFifthPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    double squareThisValue; // Variable used to store the value that wants to be squared.
	for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		squareThisValue = threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow];
		threadData->dA_u[currentRow] = 5 * squareThisValue * squareThisValue;
	}
}

// --------- RAISE TO THE 6TH POWER ACTIVATION FUNCTION ---------- //
static void getRaiseToTheSixthPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    double squareThisValue; // Variable used to store the value that wants to be squared.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
    	squareThisValue = threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow];
		threadData->A_u[currentRow] = squareThisValue * squareThisValue * squareThisValue;
	}
}
static void getDerivateRaiseToTheSixthPowerActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    double squareThisValue; // Variable used to store the value that wants to be squared.
	for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		squareThisValue = threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow];
		threadData->dA_u[currentRow] = 6 * squareThisValue * squareThisValue * threadData->f_x_tilde[currentRow];
	}
}

// ------- 1ST ORDER DEGREE EXPONENTIAL ACTIVATION FUNCTION ------ //
static void getFirstOrderDegreeExponentialActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->A_u[currentRow] = exp(threadData->f_x_tilde[currentRow]);
	}
}
static void getDerivateFirstOrderDegreeExponentialActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->dA_u[currentRow] = threadData->A_u[currentRow];
	}
}

// ------- 2ND ORDER DEGREE EXPONENTIAL ACTIVATION FUNCTION ------ //
static void getSecondOrderDegreeExponentialActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of A(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->A_u[currentRow] = exp(threadData->f_x_tilde[currentRow] * threadData->f_x_tilde[currentRow]);
	}
}
static void getDerivateSecondOrderDegreeExponentialActivation_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We calculate the part of dA(u) that corresponds to the current thread.
    for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->dA_u[currentRow] = 2 * threadData->f_x_tilde[currentRow] * threadData->A_u[currentRow];
	}
}


/**
* The "getNeuronAdjustedCoefficientOfDetermination_parallelCPUvoidPart1()"
* static function is used to apply the first part of a regression
* evaluation metric known as the adjusted coefficient of determination
* through the use of CPU parallelism. Such method will be applied with
* respect to the argument pointer variables "threadData->neuronData.Y"
* and "threadData->A_u". Then, its result will be stored in the argument
* pointer variables "threadData->accuracyTerm1" and
* "threadData->accuracyTerm2".
* 
* To learn more about the argument cpuParallelData structure variable
* that is used in this function, read the code comments that are
* located in first lines written in this file.
*
* NOTE: RESULTS ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLES
*       "threadData->accuracyTerm1" AND "threadData->accuracyTerm2".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 14, 2022
* LAST UPDATE: N/A
*/
static void getNeuronAdjustedCoefficientOfDetermination_parallelCPUvoidPart1(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;

	// We reset the "SSE" and "the sum of the output matrix" that will be collected by this particular CPU thread.
	threadData->accuracyTerm1 = 0;
	threadData->accuracyTerm2 = 0;

	// We obtain the sums required for the means to be calculated and the SSE values for each of the columns of the input matrix.
	double squareThisValue; // Variable used to store the value that wants to be squared, for performance purposes.
	double mean_realOutputMatrix = 0; // This variable is used to store the means of all the outputs of the argument pointer variable "realOutputMatrix".
	for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		// We make the required calculations to obtain the SSE values.
		squareThisValue = threadData->neuronData.Y[currentRow] - threadData->A_u[currentRow]; // real output matrix - predicted output matrix
		threadData->accuracyTerm1 += (squareThisValue * squareThisValue); // We temporarly store the SSE values in the argument pointer variable "adjustedRsquared", for performance purposes.

		// We make the required calculations to obtain the sums required for the calculation of the means.
    	threadData->accuracyTerm2 += threadData->neuronData.Y[currentRow];
	}
}
/**
* The "getNeuronAdjustedCoefficientOfDetermination_parallelCPUvoidPart2()"
* static function is used to apply the first part of a regression
* evaluation metric known as the adjusted coefficient of determination
* through the use of CPU parallelism. Such method will be applied with
* respect to the argument pointer variables "threadData->neuronData.Y"
* and "threadData->A_u". Then, its result will be stored in the argument
* pointer variables "threadData->accuracyTerm1" and
* "threadData->accuracyTerm2".
*
* To learn more about the argument cpuParallelData structure variable
* that is used in this function, read the code comments that are
* located in first lines written in this file.
*
* NOTE: RESULTS ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLES
*       "threadData->accuracyTerm1" AND "threadData->accuracyTerm2".
*
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 14, 2022
* LAST UPDATE: N/A
*/
static void *getNeuronAdjustedCoefficientOfDetermination_parallelCPUvoidPart2(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;
	
	// We obtain the SST values that will be required to make the calculation of the adjusted coefficient of determination.
	threadData->accuracyTerm1 = 0; // We reset the "SST" that will be collected by this particular CPU thread.
	double squareThisValue; // Variable used to store the value that wants to be squared, for performance purposes.
	for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		// We make the required calculations to obtain the SST values.
		squareThisValue = threadData->neuronData.Y[currentRow] - threadData->accuracyTerm2; // real output matrix - mean of real output matrix
		threadData->accuracyTerm1 += (squareThisValue * squareThisValue); // We temporarly store the SSE values in the argument pointer variable "R", for performance purposes.
	}
	
	return NULL;
}


/**
* The "predictSingleNeuronDNN_parallelCPU()" function is used to make the
* predictions of the requested input values (X) by applying the
* simple linear equation system with the specified coefficient values
* (b). The predicted values will be stored in the argument pointer
* variable "Y_hat".
* 
* @param struct singleNeuronDnnStruct_parallelCPU *neuron - This
*					 argument will contain the pointer to a struct
*					 variable that should contain all the information
*					 required in order to be able to create and make
*					 an artificial neuron. Its accessible inner
*					 elements are described in the list showed in
*					 the commented documentation of the function
*					 "getSingleNeuronDNN_parallelCPU()".
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
* CREATION DATE: JANUARY 16, 2022
* LAST UPDATE: N/A
*/
void predictSingleNeuronDNN_parallelCPU(struct singleNeuronDnnStruct_parallelCPU *neuron, double *Y_hat) {
	// If the CPU threads are less than value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->cpuThreads < 1) {
		printf("\nERROR: The requested CPU threads must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the CPU threads are greater than the number of samples, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->cpuThreads) > (neuron->n)) {
		printf("\nERROR: The requested CPU threads must be equal or less than the number of machine learning samples for this particular algorithm.\n");
		exit(1);
	}
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

	// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
	// We initialize the data that we will give to each of the CPU threads to be created.
	struct cpuParallelData threadData[neuron->cpuThreads]; // We create a struct variable used to store all the data that will be required by the CPU threads to be created.
	int mPlusOne = neuron->m + 1; // This value is repetitively used and strategically stored here for performance purposes.
	for (int currentThread; currentThread<(neuron->cpuThreads); currentThread++) {
		threadData[currentThread].neuronData = neuron[0]; // We give the neuron's input data to each CPU thread.
		threadData[currentThread].threadStart = currentThread * neuron->n / neuron->cpuThreads; // We assign the starting working point of each thread.
		threadData[currentThread].threadStop = (currentThread+1) * neuron->n / neuron->cpuThreads; // We assign the working point at which each thread will stop.
		threadData[currentThread].mPlusOne = mPlusOne; // We give the directory address of the variable that contains the value that has to be in the "mPlusOne" pointer variable.
		threadData[currentThread].f_x_tilde = Y_hat; // We pass the pointer of the variable that will store the predicted output data of the current single artificial neuron model.
		threadData[currentThread].A_u = Y_hat; // We pass the ponter of the variable that will store the predicted output data of the current single artificial neuron model.
	}

	// --------------- DATA PREDICTION PROCESS --------------- //
	// We obtain and store the transpose of "X_tilde".
	pthread_t threadId[neuron->cpuThreads]; // pthread_t object definition with an integer array structure to be used as an ID for each created thread.
	for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We create the specified threads and assign them the function they will work with.
		pthread_create(&threadId[currentThread], NULL, getPredictSingleNeuronDNN_parallelCPU, &threadData[currentThread]);
	}

	for(int currentThread=0; currentThread<(neuron->cpuThreads); currentThread++) { // We force the program to wait until all threads have finished their assigned task.
		pthread_join(threadId[currentThread], NULL);
	}

	return;
}


/**
* The "getPredictSingleNeuronDNN_parallelCPU()" static function
* is used to calculate the prediction made by a specified single
* artificial nueron model through the use of CPU parallism.
*
* To learn more about the argument cpuParallelData structure variable
* that is used in this function, read the code comments that are
* located in first lines written in this file.
*
* NOTE: RESULTS ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "threadData->f_x_tilde".
*
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 16, 2022
* LAST UPDATE: N/A
*/
static void *getPredictSingleNeuronDNN_parallelCPU(void* threadVariable) {
	// We create a structure variable to access the data that was assigned for the current CPU thread.
	struct cpuParallelData* threadData = (struct cpuParallelData*) threadVariable;

	// --------------- BEGINNING OF PREDICTION PROCESS --------------- //
	// We calculate the currently predicted output data made by the body of the neuron and store it in "f_x_tilde". However, for performance purposes, we will temporarily store the values of "f_x_tilde" in "Y_hat".
	int currentRowTimesM; // This variable is used to store a repetitive multiplication in some for-loops, for performance purposes.
	for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
		threadData->f_x_tilde[currentRow] = threadData->neuronData.w_best[0];
		currentRowTimesM = currentRow * threadData->neuronData.m;
		for (int currentColumn=1; currentColumn<(threadData->mPlusOne); currentColumn++) {
			threadData->f_x_tilde[currentRow] = threadData->f_x_tilde[currentRow] + threadData->neuronData.w_best[currentColumn] * threadData->neuronData.X[currentColumn-1 + currentRowTimesM];
		}
	}

	// We calculate, in its continous (regression) form, the currently predicted output data made by the neuron and store it in "Y_hat" by applying the desired activation function to "f_x_tilde".
	// NOTE: Remember that "Y_hat" = A(u) = "A_u".
	// NOTE: "activationFunctions" is a pointer to each of the individual activation functions that were developed as static void functions.
	(*activationFunctions[threadData->neuronData.activationFunctionToBeUsed])(threadVariable); // We calculate A(u) and store it in the pointer variable "f_x_tilde".

	// Determine if the given model of a single neuron in Deep Neural Network is meant for a classification or for a regression problem to then make the predictions accordingly.
	if (threadData->neuronData.isClassification == 1) {
		// We apply the threshold define by the implementer in order to obtain a classification output and store it in "f_x_tilde".
		for (int currentRow=(threadData->threadStart); currentRow<(threadData->threadStop); currentRow++) {
			if (threadData->f_x_tilde[currentRow] > threadData->neuronData.threshold) {
				threadData->f_x_tilde[currentRow] = threadData->neuronData.desiredValueForGroup1; // Group 1 has been predicted.
			} else {
				threadData->f_x_tilde[currentRow] = threadData->neuronData.desiredValueForGroup2; // Group 2 has been predicted.
			}
		}
	}

	return NULL;
}
