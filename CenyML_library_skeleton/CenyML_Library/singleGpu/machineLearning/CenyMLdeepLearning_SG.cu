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
#include "CenyMLdeepLearning_SG.h"
#include "../../../../CenyML_library_skeleton/otherLibraries/cuda/CUDA_check.h"
// IMPORTANT NOTE: This library uses the math.h library and therefore, remember
// 		to use the "-lm" flag when compiling it.


/**
* The "getSingleNeuronDNN_singleGPU()" function is used to apply the machine
* learning algorithm called single neuron in Deep Neural Network as formulated
* in the master thesis of Cesar Miranda Meza called "Machine learning to support
* applications with embedded systems and parallel computing", but in its single
* GPU parallel version. Within this process, the best fitting equation with the
* form of "y_hat = b_0 + w_1*x_1 + w_2*x_2 + ... + w_m*x_m" will be identified
* with respect to the sampled data given through the argument pointer variables
* "neuron->X" and "neuron->Y". As a result, the identified coefficient values
* will be stored in the argument pointer variable "neuron->w_new". With respect
* to the struct pointer variable "neuron", it should contain all the information
* required in order to be able to create and make an artificial neuron. Its
* accessible inner elements will be described in the following list:
*
*
* @param int gpuDevice - This argument will represent the desired GPU (device)
* 			with which the implementer wants this algorithm to be
* 			parallelized.
*
* @param double *X - This argument will contain the pointer to a memory
* 		allocated input matrix, from which the desired machine learning
* 		algorithm will be calculated. THIS VARIABLE SHOULD BE ALLOCATED
*		AND INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* @param double *w_first - This argument will contain the pointer to a memory
* 			allocated coefficient matrix. The use of this variable
* 			will difer depending on the value assigned in the
*			argument variable "isInitial_w", whose possible outcomes
* 			are listed below:
*			1) "isInitial_w"=(int)1 --> "w_first"
*			HAS TO BE INITIALIZED BEFORE CALLING THIS FUNCTION
* 			because its defined coefficient values will be assigned
*			to the neuron as its initial weight values before
* 			starting its training process.
*			2) "isInitial_w"=(int)0 --> "w_first"
*			does not require to be initialized but has to be
* 			allocated in memory. After this function concludes its
* 			processes, the implementer will be able to know what were
* 			the initial weight values that the neuron had when it was
*			created. Regardless of the value of "isInitial_w",
* 			"w_first" SHOULD BE ALLOCATED BEFORE CALLING THIS
* 			FUNCTION WITH A SIZE OF "1" TIMES "m+1" 'DOUBLE' MEMORY
* 			SPACES.
*
* @param double *Y - This argument will contain the pointer to a memory
* 		allocated output matrix, representing the real data of the
* 		system under study. This variable will be used as a reference to
* 		apply the desired machine learning algorithm. THIS VARIABLE
* 		SHOULD BE ALLOCATED AND INITIALIZED BEFORE CALLING THIS FUNCTION
* 		WITH A SIZE OF "n" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of samples (rows)
* 		that the input matrix has, with which the output data was
* 		obtained.
*
* @param int m - This argument will represent the total number of features
* 		(independent variables) that the input matrix has, with which
* 		the output data was obtained.
*
* @param int p - This argument will represent the total number of outputs that
* 		exist in the the output matrix, containing the real results of
* 		the system under study.
*
* @param char isInitial_w = This argument variable will work as a flag to
* 			indicate whether the coefficients contained in the
* 			argument variable "w_first" will be used as the initial
* 			weight values for the neuron to be created or not. The
* 			possible values for "isInitial_w" are the following:
*			1) "isInitial_w"=(int)1 --> the coefficient values of
* 			"w_first" will be assigned to the neuron as its initial
* 			weight values before starting its training process.
*			2) "isInitial_w"=(int)0 --> the coefficient values of
* 			"w_first" will not be assigned to the neuron as its
* 			initial weight values and after having called this
* 			function, the implementer will be able to retrieve from
* 			"w_first" the coefficient values with which the neuron
*			had been created before starting its learning process.
*
* @param char isClassification = This argument variable will work as a flag to
* 				indicate to the neron if it is expected from it
* 				to interpret the given data of "X" and "Y" as if
* 				their were meant for a classification problem or
* 				not. The possible valid values for this flag are
* 				the following:
*				1) "isClassification" = (int) 1 --> The neuron
* 				will interpret the data of "X" and "Y" as if they
* 				were meant for a classification problem.
*				2) "isClassification" = (int) 0 --> The neuron
* 				will interpret the data of "X" and "Y" as if they
* 				were meant for a regression problem.
*
* @param double threshold - This argument will represent desired threshold that
* 			the implementer desired the neuron to consider in
* 			classification problems. In this regard, whenever the
* 			predicted output of the neuron is higher than the
* 			defined threshold value, then that prediction should be
*			interpreted as group 1 (ussually refered to as the binary
* 			output 1). Conversely, if the predicted value is lower
* 			than the defined threshold value, then that prediction
* 			should be interpreted as group 2 (ussually refered to as
*			the binary output 0). However, have in mind that
* 			"threshold" will only be used by the neuron if the
* 			argument variable "isClassification" = 1.
*
* @param int desiredValueForGroup1 - This argument will represent the desired
*				label value to whenever an output of the neuron
* 				predicts the classification group 1. Ussually,
* 				this is label with the value of "(int) 1" but any
* 				other customized value can be assigned by the
* 				implementer. However, have in mind that this
* 				argument variable will be considered by the
* 				neuron as long as the argument variable
*				"isClassification" = 1 and only when the
*				implementer requests to the neuron a prediction
* 				through the function
* 				"predictSingleNeuronDNN_singleGPU()".
*
* @param int desiredValueForGroup2 - This argument will represent the desired
*				label value to whenever an output of the neuron
* 				predicts the classification group 2. Ussually,
* 				this is label with the value of "(int) -1" but
* 				any other customized value can be assigned by the
* 				implementer. However, have in mind that this
* 				argument variable will be considered by the
* 				neuron as long as the argument variable
*				"isClassification" = 1 and only when the
*				implementer requests to the neuron a prediction
* 				through the function
* 				"predictSingleNeuronDNN_singleGPU()".
*
* @param int activationFunctionToBeUsed - This argument will represent the
* 					identifier of the desired activation
* 					function to be used by the neuron during
* 					its training process. Its possible valid
* 					values are the following:
*					0 = Rectified Linear Units (ReLU).
*					1 = Hyperbolic tangent (tanh).
*					2 = Logistic function.
*					3 = Raise to the 1st power.
*					4 = Raise to the 2nd power.
*					5 = Raise to the 3rd power.
*					6 = Raise to the 4th power.
*					7 = Raise to the 5th power.
*					8 = Raise to the 6th power.
*					9 = 1st order degree exponential.
*					10 = 2nd order degree exponential.
*
* @param double learningRate - This argument will represent the hyperparameter
* 			value known as the learning rate of the artificial
* 			neuron. Note that there is no way to know what is going
* 			to be the best learning rate value for your particular
* 			problem to be solved by the neuron because the best one
* 			differs from one problem to another. Therefore, you will
* 			most likely have to experiment with several values until
* 			you find the model solution that satisfies you the most.
*
* @param double stopAboveThisAccuracy - This argument will represent a a stop
* 				value for the training process. The way this
* 				value will work is that if the neuron gets an
* 				evaluation metric result that is strictly higher
* 				than the one defined in "stopAboveThisAccuracy",
* 				then the neuron will stop its training process
*				and this function will end. Note that the
* 				evaluation metric to be used will be the adjusted
* 				R squared regardless if the data is for
* 				classification or not.
*
* @param int maxEpochs - This argument will represent the maximum number of
* 			epochs that are desired for the training process of the
*			artificial neuron. Note that for each epoch that occurs,
*			that should be interpreted as the neuron having updated
*			its weight values one time.
*
* @param char isReportLearningProgress = This argument variable will work as a
*				flag to indicate to the neuron if it is desired
* 				that it reports its learning progress to the
* 				user. The following will list the possible valid
* 				outcomes for this variable:
*				1) "isReportLearningProgress" = (int) 1:
*				The neuron will interpret this as being
* 				instructed to report its learning progress to the
* 				user through the window terminal by displaying
*				messages over time.
*				2) "isReportLearningProgress" = (int) 0:
*				The neuron will interpret this as being
* 				instructed not to report its learning progress.
*
* @param int reportEachSpecifiedEpochs - This argument variable will indicate
*				how many each amount of epochs it is desired by
* 				the implementer that the artificial neuron
* 				reports its learning progress to the user.
* 				However, in order for the neuron to consider this
* 				variable, it will be strictly needed to set the
*				argument variable "isReportLearningProgress" =
* 				(int) 1.
*
* @param double *w_best - This argument will contain the pointer to a memory
*			allocated variable in which we will store the identified
* 			best fitting coefficient values for the model of a single
* 			neuron in Deep Neural Network. These coefficients will
* 			each be stored in the same row but under different
* 			columns where the first coefficient (b_0) will be stored
* 			in the column with index 0; the second coefficient (w_1)
* 			will be stored in the column index 1 and; the last
* 			coefficient (w_m) will be stored in the column index m.
* 			IT IS INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
* 			BEFORE CALLING THIS FUNCTION WITH A VARIABLE SIZE OF "1"
* 			TIMES "m+1" 'DOUBLE' MEMORY SPACES.
*
* @param double bestAccuracy - This argument will contain the value of the best
*			accuracy that the neuron was able to achieve during its
* 			training process.
*
* @param double *w_new - This argument will contain the pointer to a memory
*			allocated variable in which we will store the last
*			identified coefficient values for the model of a single
* 			neuron in Deep Neural Network. These coefficients will
* 			each be stored in the same row but under different
* 			columns where the first coefficient (b_0) will be stored
* 			in the column with index 0; the second coefficient (w_1)
* 			will be stored in the column index 1 and; the last
* 			coefficient (w_m) will be stored in the column index m.
* 			IT IS INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
* 			BEFORE CALLING THIS FUNCTION WITH A VARIABLE SIZE OF "1"
* 			TIMES "m+1" 'DOUBLE' MEMORY SPACES.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "w_best" that
* 	is contained in the struct pointer variable "neuron".
*
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 24, 2022
* LAST UPDATE: JANUARY 25, 2022
*/
void getSingleNeuronDNN_singleGPU(struct singleNeuronDnnStruct_singleGPU *neuron) {
	// If the requested GPU (device) is less than zero, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->gpuDevice < 0) {
		printf("\nERROR: The identifier of the requested GPU (device) must be equal or greater than 0.\n");
		exit(1);
	}
	// If the value of "neuron->maxUnrollingLoop" is not in the range of 1 and 10, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->maxUnrollingLoop<1) && (neuron->maxUnrollingLoop>10)) {
		printf("\nERROR: The defined value for \"maxUnrollingLoop\" in the struct of \"singleNeuronDnnStruct\" that you created can only have a whole value in the range of 1 and 10. Please add a valid value to it.\n");
		exit(1);
	}
	// If the machine learning samples are less than value of two, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->n < 2) {
		printf("\nERROR: The machine learning samples must be equal or greater than 2 for this particular algorithm.\n");
		exit(1);
	}
	// If the machine learning features are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->m < 1) {
		printf("\nERROR: The machine learning features (independent variables) must be equal or greater than 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->p != 1) {
		printf("\nERROR: The outputs of the system under study must be equal to 1 for this particular algorithm.\n");
		exit(1);
	}
	// If the identifier assigned to "neuron->activationFunctionToBeUsed" is not in the range of 0 and 11, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->activationFunctionToBeUsed>11) && (neuron->activationFunctionToBeUsed<0)) {
		printf("\nERROR: The defined activation function identifier assigned to \"activationFunctionToBeUsed\" in the struct of \"singleNeuronDnnStruct\" that you created must be a whole value in the range of 0 to 11. Please add a valid identifier number to it.\n");
		exit(1);
	}
	// If the flag "neuron->isClassification" has a value different of 0 and 1, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->isClassification!=0) && (neuron->isClassification!=1)) {
		printf("\nERROR: The defined value for the flag \"isClassification\" in the struct of \"singleNeuronDnnStruct\" that you created can only have a value of either 0 or 1. Please add a valid value to it.\n");
		exit(1);
	}
	// If the flag "neuron->isInitial_w" has a value different of 0 and 1, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->isInitial_w!=1) && (neuron->isInitial_w!=0)) {
		printf("\nERROR: The defined value for the flag \"isInitial_w\" in the struct of \"singleNeuronDnnStruct\" that you created can only have a value of either 0 or 1. Please add a valid value to it.\n");
		exit(1);
	}
	// If the flag "neuron->isReportLearningProgress" has a value different of 0 and 1, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->isReportLearningProgress!=1) && (neuron->isReportLearningProgress!=0)) {
		printf("\nERROR: The defined value for the flag \"isReportLearningProgress\" in the struct of \"singleNeuronDnnStruct\" that you created can only have a value of either 0 or 1. Please add a valid value to it.\n");
		exit(1);
	}
	// If the requested epochs are less than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->maxEpochs < 1) {
		printf("\nERROR: The defined value for \"maxEpochs\" in the struct of \"singleNeuronDnnStruct\" that you created must be equal or greater than 1 for this particular algorithm. Please add a valid value to it.\n");
		exit(1);
	}
	// If the "neuron->reportEachSpecifiedEpochs" is less than one and greater than "neuron->maxEpochs", then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->reportEachSpecifiedEpochs<1) && (neuron->maxEpochs<neuron->reportEachSpecifiedEpochs)) {
		printf("\nERROR: The defined value for \"reportEachSpecifiedEpochs\" in the struct of \"singleNeuronDnnStruct\" that you created cannot be less than 1 and cannot be greater than the value of \"maxEpochs\" contained in such struct. Please add a valid value to it.\n");
		exit(1);
	}
	// If the value of "neuron->stopAboveThisAccuracy" is not in the range of 0 and 1, then emit an error message and terminate the program. Otherwise, continue with the program.
	if ((neuron->stopAboveThisAccuracy<0) && (neuron->stopAboveThisAccuracy>1)) {
		printf("\nERROR: The defined value for the flag \"stopAboveThisAccuracy\" in the struct of \"singleNeuronDnnStruct\" that you created can only have a value in the range of 0 and 1. Please add a valid value to it.\n");
		exit(1);
	}
	
	
	// ------- SELECTION AND INITIALIZATION OF THE DESIRED GPU ------- //
	// We select the desired GPU by the implementer and inform in the terminal the name of such GPU.
	cudaDeviceProp gpuProperties;
	CHECK(cudaGetDeviceProperties(&gpuProperties, neuron->gpuDevice)); // We obtain the details of the GPU that was defined by the implementer.
	printf("\nThe GPU (device) %d: %s, has been selected by the CenyML library.\n", neuron->gpuDevice, gpuProperties.name);
	CHECK(cudaSetDevice(neuron->gpuDevice)); // We select the GPU that was requested by the implementer.
	
	// Set up the execution configurations that will be assigned to the selected GPU.
	dim3 block_32x_1y(32, 1); // We define the number of GPU threads per block.
	dim3 grid_n((neuron->n + block_32x_1y.x - 1) / block_32x_1y.x, 1); // We define the number of blocks that our GPU will manage when the number of samples is "neuron->n".
	
	// We determine what number of Unrolling Loop strategy will be applied according to the number of samples and the value of "neuron->maxUnrollingLoop" that were given by the implementer/user.
	int numberOfUnrollingLoop1; // Variable used to store the number of unrolling loops that the algorithm will use in the first Unrolling Parallel Reduction strategy that will be applied for each case.
	for (numberOfUnrollingLoop1=neuron->maxUnrollingLoop; numberOfUnrollingLoop1>0; numberOfUnrollingLoop1--) {
		// NOTE: The idea in this for-loop is to find the highest number, up to a maximum of 10, that can completely divide the number
		//	  of blocks defined for the selected GPU. However, because the process of defining the number of blocks is conveniently
		//	  automated for performance purposes, the implementer can attempt to achieve the highest or a higher unrolling loop if
		//	  he changes the number of input samples given to this function.
		if (grid_n.x%numberOfUnrollingLoop1 == 0) {
			if (numberOfUnrollingLoop1 == 1) {
				printf("This algorithm WILL NOT apply the \"Unrolling Loop Strategy\" due to the number of samples given and/or the defined maximum unrolling loop.\n");
			} else {
				printf("This algorithm will apply the \"Unrolling%d Loop Strategy\" for each case applicable (the current maximum limit is %d).\n", numberOfUnrollingLoop1, neuron->maxUnrollingLoop);
			}
			break;
		}
	}
	int trueUnrollingSize1 = grid_n.x/numberOfUnrollingLoop1; // This variable is used to store the grid size that will be considered for all the processes that apply the first Unrolling Parallel Reduction strategy for each case, for performance purposes.
	
	// We configure the shared memory of the current GPU.
	cudaSharedMemConfig pConfig = cudaSharedMemBankSizeEightByte; // We create a cudaSharedMemConfig type variable to store in it the configuration of 8-byte mode for shared memory in the GPU.
	cudaDeviceSetSharedMemConfig(pConfig); // We set the 8-byte mode for shared memory in the selected GPU.
	
	// We create the pointers to the data that the selected GPU will require.
	int mPlusOne = neuron->m + 1; // This value is repetitively used and strategically stored here for performance purposes.
	double *d_X; // This pointer variable is used to store the data from "neuron->X" into the selected GPU.
	double *d_Y; // This pointer variable is used to store the data from "neuron->Y" into the selected GPU.
	double *d_w_new; // This pointer variable is used to store the data from "neuron->w_new" into the selected GPU.
	double *d_TransposeOf_X_tilde; // This pointer variable is used to store the transpose of the transformed version of "d_X" into "X_tilde" in the selected GPU.
	double *d_f_x_tilde; // This pointer variable is used to store the output of the body of the neuron in the selected GPU.
	double *d_A_u; // This pointer variable is used to store the output of the application of the chosen activation function in the selected GPU.
	double *d_dA_u; // This pointer variable is used to store the output of the application of the derivative of the chosen activation function in the selected GPU.
	double *d_accuracyTerm1; // This pointer variable is used to store key data that is required to calculate the adjusted R squared of the model generated by the neuron.
	double *d_accuracyTerm2; // This pointer variable is used to store key data that is required to calculate the adjusted R squared of the model generated by the neuron.
	double *d_reducedAccuracyTerm1; // This pointer variable is used to store the data of "d_accuracyTerm1", but after having applied the Parallel Reduction Strategy.
	double *d_reducedAccuracyTerm2; // This pointer variable is used to store the data of "d_accuracyTerm2", but after having applied the Parallel Reduction Strategy.
	double *d_errorTerm; // This pointer variable is used to store key data for the calculation of the error term that is applied in the learning process of the artificial neuron.
	double *d_errorTerm_dot_Xtilde; // This pointer variable is used to store the data of "d_errorTerm", but after having applied the Parallel Reduction Strategy.
	
	// We allocate the required memory in the selected GPU.
	CHECK(cudaMalloc((void **) &d_X, neuron->n*neuron->m*sizeof(double)));
	CHECK(cudaMalloc((void **) &d_Y, neuron->n*sizeof(double)));
	int w_new_Bytes = mPlusOne*sizeof(double); // This variable stores the number of bytes to allocate the new weight values that will be obtained each epoch of the training process, for performance purposes.
	CHECK(cudaMalloc((void **) &d_w_new, w_new_Bytes));
	CHECK(cudaMalloc((void **) &d_TransposeOf_X_tilde, mPlusOne*neuron->n*sizeof(double)));
	CHECK(cudaMalloc((void **) &d_f_x_tilde, neuron->n*sizeof(double)));
	CHECK(cudaMalloc((void **) &d_A_u, neuron->n*sizeof(double)));
	CHECK(cudaMalloc((void **) &d_dA_u, neuron->n*sizeof(double)));
	CHECK(cudaMalloc((void **) &d_accuracyTerm1, neuron->n*sizeof(double)));
	CHECK(cudaMalloc((void **) &d_accuracyTerm2, neuron->n*sizeof(double)));
	int parRed_Bytes = trueUnrollingSize1*sizeof(double); // This variable stores the number of bytes to allocate in those variables that will have the Unrolling Parallel Reduction strategy applied once, for performance purposes.
	CHECK(cudaMalloc((void **) &d_reducedAccuracyTerm1, parRed_Bytes));
	CHECK(cudaMalloc((void **) &d_reducedAccuracyTerm2, parRed_Bytes));
	CHECK(cudaMalloc((void **) &d_errorTerm, mPlusOne*neuron->n*sizeof(double)));
	int errorTerm_dot_Xtilde_Bytes = mPlusOne*trueUnrollingSize1*sizeof(double); // This variable stores the number of bytes to allocate the device variable "d_errorTerm_dot_Xtilde", for performance purposes.
	CHECK(cudaMalloc((void **) &d_errorTerm_dot_Xtilde, errorTerm_dot_Xtilde_Bytes));
	
	
	// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
	// We transfer the input data that the neuron will need into the selected GPU.
	CHECK(cudaMemcpy(d_X, neuron->X, (neuron->n*neuron->m*sizeof(double)), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_Y, neuron->Y, (neuron->n*sizeof(double)), cudaMemcpyHostToDevice));
	
	// We obtain the transpose of "X_tilde" in the GPU.
	getTransposeOfInputData_singleGPU <<< grid_n, block_32x_1y >>> (d_X, neuron->n, mPlusOne, d_TransposeOf_X_tilde);
	
	
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
	
	// We pass the generated weights to the GPU.
	CHECK(cudaMemcpy(d_w_new, neuron->w_new, w_new_Bytes, cudaMemcpyHostToDevice));
	
	// We allocate all the memory that will be required in the CPU for the training process of the neuron.
	int double32_Bytes = 32*sizeof(double); // This variable stores the number of bytes required to store 32 double variable type, for performance purposes.
	int double64_Bytes = 2*double32_Bytes; // This variable stores the number of bytes required to store 64 double variable type, for performance purposes.
	int double96_Bytes = 3*double32_Bytes; // This variable stores the number of bytes required to store 96 double variable type, for performance purposes.
	int double128_Bytes = 2*double64_Bytes; // This variable stores the number of bytes required to store 128 double variable type, for performance purposes.
	int unrollingTotalBlockSize1 = grid_n.x * block_32x_1y.x; // This variable stores the total number of GPU threads that are to be employed specifically for calling a GPU Kernel to apply the Parallel Reduction strategy to a certain set of data. NOTE: This variable has more application/sense when applying the Parallel Reduction two or more consecutive times.
	double *h_reducedAccuracyTerm1 = (double *) malloc(parRed_Bytes); // CPU Allocated variable that will contain all the individual contributions made by each thread block in an attemp to apply the parallel reduction strategy to "d_accuracyTerm1".
	double *h_reducedAccuracyTerm2 = (double *) malloc(parRed_Bytes); // CPU Allocated variable that will contain all the individual contributions made by each thread block in an attemp to apply the parallel reduction strategy to "d_accuracyTerm2".
	double totalSumOfAccuracyTerm1 = 0; // This variable is used to sequentially sum all the contributions of each GPU block that were made to get "d_accuracyTerm1" and that were stored in "h_reducedAccuracyTerm1".
	double totalSumOfAccuracyTerm2 = 0; // This variable is used to sequentially sum all the contributions of each GPU block that were made to get "d_accuracyTerm2" and that were stored in "h_reducedAccuracyTerm2".
	int nMinusOne = neuron->n-1; // This variable is used to store a repetitive value that is used several times in the program, for performance purposes.
	double currentAccuracy = 0; // This variable is used to contain the current accuracy of the neuron.
	double *idata; // This variable is used to convert a pointer of interest to have a new origin from such pointer.
	double *h_errorTerm_dot_Xtilde = (double *) malloc(errorTerm_dot_Xtilde_Bytes); // CPU allocated variable that will contain all the individual contributions made by each thread block in an attemp to apply the parallel reduction strategy to "d_errorTerm".
	double totalErrorTerm_dot_Xtilde = 0; // This variable is used to sum all the contributions of each GPU block that were made to get "d_errorTerm" and that were stored in "d_errorTerm_dot_Xtilde".
	double *w_old = (double *) malloc(w_new_Bytes); // Allocate the memory required for the variable "w_old", which will contain the previous weight values that were obtained with respect to the current ones.
	
	
	// ------------------------------------- //
	// ----- REGRESSION MODEL SELECTED ----- //
	// ------------------------------------- //
	
	// ----------- EVALUATION OF THE INITIAL WEIGHT VALUES ----------- //
	// We calculate "f_x_tilde", "A(u)", "dA(u)" and "the part 1 of the accuracy terms".
	getFxTilde_Au_dAu_and_accuracyTermsPart1_singleGPU <<< grid_n, block_32x_1y, double96_Bytes >>> (d_X, d_Y, d_w_new, neuron->n, neuron->m, neuron->activationFunctionToBeUsed, d_f_x_tilde, d_A_u, d_dA_u, d_accuracyTerm1, d_accuracyTerm2);
	CHECK(cudaDeviceSynchronize()); // We force the program to wait until all GPU threads have finished the last task they were given.
	getParallelReduction <<< trueUnrollingSize1, block_32x_1y, double32_Bytes >>> (d_accuracyTerm1, d_reducedAccuracyTerm1, unrollingTotalBlockSize1, numberOfUnrollingLoop1); // We apply the parallel reduction strategy on "d_accuracyTerm1".
	getParallelReduction <<< trueUnrollingSize1, block_32x_1y, double32_Bytes >>> (d_accuracyTerm2, d_reducedAccuracyTerm2, unrollingTotalBlockSize1, numberOfUnrollingLoop1); // We apply the parallel reduction strategy on "d_accuracyTerm2".
	
	// We calculate the sequential part of "the part 1 of the accuracy terms" by sequentially summing all the contributions made and stored in "d_reducedAccuracyTerm1" and "d_reducedAccuracyTerm2" after having applied the parallel reduction strategy on them.
	CHECK(cudaMemcpy(h_reducedAccuracyTerm1, d_reducedAccuracyTerm1, parRed_Bytes, cudaMemcpyDeviceToHost)); // We transfer the GPU data from "d_reducedAccuracyTerm1" to the CPU through "h_reducedAccuracyTerm1".
	CHECK(cudaMemcpy(h_reducedAccuracyTerm2, d_reducedAccuracyTerm2, parRed_Bytes, cudaMemcpyDeviceToHost)); // We transfer the GPU data from "d_reducedAccuracyTerm2" to the CPU through "h_reducedAccuracyTerm2".
	totalSumOfAccuracyTerm1 = 0; // We reset the value of the accuracy term 1, in which we will store the value of SSE.
	totalSumOfAccuracyTerm2 = 0; // We reset the value of the accuracy term 2, in which we will temporarily store the sum of all the values from the "real output matrix".
	for (int currentBlock=0; currentBlock<trueUnrollingSize1; currentBlock++) {
		totalSumOfAccuracyTerm1 += h_reducedAccuracyTerm1[currentBlock]; // We sum all the fragments of the SSE that was calculated by the previous parallelization process.
		totalSumOfAccuracyTerm2 += h_reducedAccuracyTerm2[currentBlock]; // We sum all the fragments of the "real output matrix sum" that was calculated by the previous parallelization process.
	}
	h_reducedAccuracyTerm2[0] = totalSumOfAccuracyTerm2 / neuron->n; // We calculate the mean of the values contained in the "real output matrix".
	
	// We calculate "the part 2 of the accuracy terms".
	CHECK(cudaMemcpy(d_reducedAccuracyTerm2, h_reducedAccuracyTerm2, sizeof(double), cudaMemcpyHostToDevice)); // We pass mean of the "real output matrix" to the GPU, which is contained in the first data location of the pointer variable "h_reducedAccuracyTerm2".
	getNeuronAdjustedCoefficientOfDetermination_singleGPUvoidPart2 <<< grid_n, block_32x_1y, double64_Bytes >>> (d_Y, neuron->n, d_accuracyTerm1, d_reducedAccuracyTerm2);
	CHECK(cudaDeviceSynchronize()); // We force the program to wait until all GPU threads have finished the last task they were given.
	getParallelReduction <<< trueUnrollingSize1, block_32x_1y, double32_Bytes >>> (d_accuracyTerm1, d_reducedAccuracyTerm1, unrollingTotalBlockSize1, numberOfUnrollingLoop1); // We apply the parallel reduction strategy on "d_accuracyTerm1", containing the SST data.
	CHECK(cudaMemcpy(h_reducedAccuracyTerm1, d_reducedAccuracyTerm1, parRed_Bytes, cudaMemcpyDeviceToHost)); // We transfer the GPU data from "d_reducedAccuracyTerm1" to the CPU through "h_reducedAccuracyTerm1".
	totalSumOfAccuracyTerm2 = 0; // We reset the value of the accuracy term 2, in which we will store the value of SST.
	for (int currentBlock=0; currentBlock<trueUnrollingSize1; currentBlock++) {
		totalSumOfAccuracyTerm2 += h_reducedAccuracyTerm1[currentBlock]; // We sum all the fragments of the SST that was calculated by the previous parallelization process.
	}

	// Finally, we calculate the adjusted coefficient of determination and store its results in the variable "currentAccuracy".
	currentAccuracy = 1 - ( (totalSumOfAccuracyTerm1/(nMinusOne-(neuron->m)))/(totalSumOfAccuracyTerm2 / nMinusOne) );
	
	// We pass the current accuracy to the best accuracy record because this is the evaluation of the very first weight values.
	neuron->bestAccuracy = currentAccuracy;
	
	// If the desired accuracy has been reached, then conclude the training process of the neuron. Otherwise, continue training it.
	if (currentAccuracy > neuron->stopAboveThisAccuracy) {
		printf("\nThe adjusted R squared (%f) of the neuron has achieved a higher one with respect to the one that was specified as a goal the very first instant it was created.\n", currentAccuracy);
		
		// Before terminating this function, we free the GPU and CPU allocated memory since they will no longer be used.
		CHECK(cudaFree(d_X));
		CHECK(cudaFree(d_Y));
		CHECK(cudaFree(d_w_new));
		CHECK(cudaFree(d_TransposeOf_X_tilde));
		CHECK(cudaFree(d_f_x_tilde));
		CHECK(cudaFree(d_A_u));
		CHECK(cudaFree(d_dA_u));
		CHECK(cudaFree(d_accuracyTerm1));
		CHECK(cudaFree(d_accuracyTerm2));
		CHECK(cudaFree(d_reducedAccuracyTerm1));
		CHECK(cudaFree(d_reducedAccuracyTerm2));
		CHECK(cudaFree(d_errorTerm));
		CHECK(cudaFree(d_errorTerm_dot_Xtilde));
		free(h_reducedAccuracyTerm1);
		free(h_reducedAccuracyTerm2);
		free(h_errorTerm_dot_Xtilde);
		free(w_old);
		return;
	}
	
	// -------- BEGINNING OF THE EPOCHS OF THE MODEL ------- //
	for (int currentEpoch=0; currentEpoch<(neuron->maxEpochs); currentEpoch++) {
		// Pass the data of "neuron->w_new" to "w_old".
		for (int currentCoefficient=0; currentCoefficient<mPlusOne; currentCoefficient++) {
			w_old[currentCoefficient] = neuron->w_new[currentCoefficient];
		}
		
		// Calculate the error term obtainable with the current weight values so that we can later update the current weight values ("w_old") in order to obtain the new ones ("neuron->w_new").
		getErrorAndUpdateWeightValues_singleGPUpart1 <<< grid_n, block_32x_1y, double128_Bytes >>> (d_TransposeOf_X_tilde, d_Y, neuron->n, mPlusOne, d_A_u, d_dA_u, d_errorTerm);
		CHECK(cudaDeviceSynchronize()); // We force the program to wait until all GPU threads have finished the last task they were given.
		getErrorAndUpdateWeightValues_singleGPUpart2 <<< trueUnrollingSize1, block_32x_1y, double32_Bytes >>> (d_errorTerm, neuron->n, mPlusOne, trueUnrollingSize1, unrollingTotalBlockSize1, numberOfUnrollingLoop1, d_errorTerm_dot_Xtilde);
		CHECK(cudaMemcpy(h_errorTerm_dot_Xtilde, d_errorTerm_dot_Xtilde, errorTerm_dot_Xtilde_Bytes, cudaMemcpyDeviceToHost)); // We transfer the GPU data from "d_errorTerm_dot_Xtilde" to the CPU through "h_errorTerm_dot_Xtilde".
		
		// We update the current weight values ("w_old") in order to obtain the new ones ("neuron->w_new") by sequentially summing all the individual contributions made after having applied the parallel reduction strategy on "d_errorTerm", whose result was stored in "h_errorTerm_dot_Xtilde".
		idata = h_errorTerm_dot_Xtilde; // We convert the pointer of interest from "h_errorTerm_dot_Xtilde" to be the origin pointer of "idata".
		for (int currentRow=0; currentRow<mPlusOne; currentRow++) {
			totalErrorTerm_dot_Xtilde = 0; // We reset the value of "totalErrorTerm_dot_Xtilde" to sum the contributed error values for the next weight.
			for (int currentBlock=0; currentBlock<trueUnrollingSize1; currentBlock++) {
				totalErrorTerm_dot_Xtilde += idata[currentBlock];
			}
			neuron->w_new[currentRow] = w_old[currentRow] + neuron->learningRate * totalErrorTerm_dot_Xtilde; // We update the current weight value.
			idata += trueUnrollingSize1; // We mode the pointer of "h_errorTerm_dot_Xtilde" to the next row/weight.
		}
		CHECK(cudaMemcpy(d_w_new, neuron->w_new, w_new_Bytes, cudaMemcpyHostToDevice)); // We pass the values of "neuron->w_new" to the GPU, through its pointer variable "d_w_new".
		
		// We recalculate "f_x_tilde", "A(u)", "dA(u)" and "the part 1 of the accuracy terms".
		getFxTilde_Au_dAu_and_accuracyTermsPart1_singleGPU <<< grid_n, block_32x_1y, double96_Bytes >>> (d_X, d_Y, d_w_new, neuron->n, neuron->m, neuron->activationFunctionToBeUsed, d_f_x_tilde, d_A_u, d_dA_u, d_accuracyTerm1, d_accuracyTerm2);
		CHECK(cudaDeviceSynchronize()); // We force the program to wait until all GPU threads have finished the last task they were given.
		getParallelReduction <<< trueUnrollingSize1, block_32x_1y, double32_Bytes >>> (d_accuracyTerm1, d_reducedAccuracyTerm1, unrollingTotalBlockSize1, numberOfUnrollingLoop1); // We apply the parallel reduction strategy on "d_accuracyTerm1".
		getParallelReduction <<< trueUnrollingSize1, block_32x_1y, double32_Bytes >>> (d_accuracyTerm2, d_reducedAccuracyTerm2, unrollingTotalBlockSize1, numberOfUnrollingLoop1); // We apply the parallel reduction strategy on "d_accuracyTerm2".
		
		// We recalculate the sequential part of "the part 1 of the accuracy terms" by sequentially summing all the contributions made and stored in "d_reducedAccuracyTerm1" and "d_reducedAccuracyTerm2" after having applied the parallel reduction strategy on them.
		CHECK(cudaMemcpy(h_reducedAccuracyTerm1, d_reducedAccuracyTerm1, parRed_Bytes, cudaMemcpyDeviceToHost)); // We transfer the GPU data from "d_reducedAccuracyTerm1" to the CPU through "h_reducedAccuracyTerm1".
		CHECK(cudaMemcpy(h_reducedAccuracyTerm2, d_reducedAccuracyTerm2, parRed_Bytes, cudaMemcpyDeviceToHost)); // We transfer the GPU data from "d_reducedAccuracyTerm2" to the CPU through "h_reducedAccuracyTerm2".
		totalSumOfAccuracyTerm1 = 0; // We reset the value of the accuracy term 1, in which we will store the value of SSE.
		totalSumOfAccuracyTerm2 = 0; // We reset the value of the accuracy term 2, in which we will temporarily store the sum of all the values from the "real output matrix".
		for (int currentBlock=0; currentBlock<trueUnrollingSize1; currentBlock++) {
			totalSumOfAccuracyTerm1 += h_reducedAccuracyTerm1[currentBlock]; // We sum all the fragments of the SSE that was calculated by the previous parallelization process.
			totalSumOfAccuracyTerm2 += h_reducedAccuracyTerm2[currentBlock]; // We sum all the fragments of the "real output matrix sum" that was calculated by the previous parallelization process.
		}
		h_reducedAccuracyTerm2[0] = totalSumOfAccuracyTerm2 / neuron->n; // We calculate the mean of the values contained in the "real output matrix".
		
		// We recalculate "the part 2 of the accuracy terms".
		CHECK(cudaMemcpy(d_reducedAccuracyTerm2, h_reducedAccuracyTerm2, sizeof(double), cudaMemcpyHostToDevice)); // We pass mean of the "real output matrix" to the GPU, which is contained in the first data location of the pointer variable "h_reducedAccuracyTerm2".
		getNeuronAdjustedCoefficientOfDetermination_singleGPUvoidPart2 <<< grid_n, block_32x_1y, double64_Bytes >>> (d_Y, neuron->n, d_accuracyTerm1, d_reducedAccuracyTerm2);
		CHECK(cudaDeviceSynchronize()); // We force the program to wait until all GPU threads have finished the last task they were given.
		getParallelReduction <<< trueUnrollingSize1, block_32x_1y, double32_Bytes >>> (d_accuracyTerm1, d_reducedAccuracyTerm1, unrollingTotalBlockSize1, numberOfUnrollingLoop1); // We apply the parallel reduction strategy on "d_accuracyTerm1", containing the SST data.
		CHECK(cudaMemcpy(h_reducedAccuracyTerm1, d_reducedAccuracyTerm1, parRed_Bytes, cudaMemcpyDeviceToHost)); // We transfer the GPU data from "d_reducedAccuracyTerm1" to the CPU through "h_reducedAccuracyTerm1".
		totalSumOfAccuracyTerm2 = 0; // We reset the value of the accuracy term 2, in which we will store the value of SST.
		for (int currentBlock=0; currentBlock<trueUnrollingSize1; currentBlock++) {
			totalSumOfAccuracyTerm2 += h_reducedAccuracyTerm1[currentBlock]; // We sum all the fragments of the SST that was calculated by the previous parallelization process.
		}

		// Finally, we recalculate the adjusted coefficient of determination and store its results in the variable "currentAccuracy".
		currentAccuracy = 1 - ( (totalSumOfAccuracyTerm1/(nMinusOne-(neuron->m)))/(totalSumOfAccuracyTerm2 / nMinusOne) );
		
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
			
			// Before terminating this function, we free the GPU and CPU allocated memory since they will no longer be used.
			CHECK(cudaFree(d_X));
			CHECK(cudaFree(d_Y));
			CHECK(cudaFree(d_w_new));
			CHECK(cudaFree(d_TransposeOf_X_tilde));
			CHECK(cudaFree(d_f_x_tilde));
			CHECK(cudaFree(d_A_u));
			CHECK(cudaFree(d_dA_u));
			CHECK(cudaFree(d_accuracyTerm1));
			CHECK(cudaFree(d_accuracyTerm2));
			CHECK(cudaFree(d_reducedAccuracyTerm1));
			CHECK(cudaFree(d_reducedAccuracyTerm2));
			CHECK(cudaFree(d_errorTerm));
			CHECK(cudaFree(d_errorTerm_dot_Xtilde));
			free(h_reducedAccuracyTerm1);
			free(h_reducedAccuracyTerm2);
			free(h_errorTerm_dot_Xtilde);
			free(w_old);
			return;
		}
	}
	
	// Determine whether it was requested that the neuron reports its learning progress or not.
	if (neuron->isReportLearningProgress == 1) { // If the implementer requested the neuron to report its progress, apply the following code.
		// Make the neuron report its last progress made.
		printf("\nEpoch %d --> single neuron in DNN has achieved an adjusted R squared of %f\n", neuron->maxEpochs, currentAccuracy);
	}
	
	// Before terminating this function, we free the GPU and CPU allocated memory since they will no longer be used.
	CHECK(cudaFree(d_X));
	CHECK(cudaFree(d_Y));
	CHECK(cudaFree(d_w_new));
	CHECK(cudaFree(d_TransposeOf_X_tilde));
	CHECK(cudaFree(d_f_x_tilde));
	CHECK(cudaFree(d_A_u));
	CHECK(cudaFree(d_dA_u));
	CHECK(cudaFree(d_accuracyTerm1));
	CHECK(cudaFree(d_accuracyTerm2));
	CHECK(cudaFree(d_reducedAccuracyTerm1));
	CHECK(cudaFree(d_reducedAccuracyTerm2));
	CHECK(cudaFree(d_errorTerm));
	CHECK(cudaFree(d_errorTerm_dot_Xtilde));
	free(h_reducedAccuracyTerm1);
	free(h_reducedAccuracyTerm2);
	free(h_errorTerm_dot_Xtilde);
	free(w_old);
	
	printf("\nThe best adjusted R squared (%f) achieved by the neuron did not surpased the defined goal but its training process has been successfully concluded.\n", neuron->bestAccuracy);
	return;
}


/**
* The "getTransposeOfInputData_singleGPU()" global static function is used to
* apply a single GPU to calculate and store the transpose of the input matrix in
* its transformed form of "X_tilde", which will be used to train a single
* artificial neuron.
* 
* 
* @param double *X - This argument will contain the pointer to a memory
*		allocated input matrix, from which the desired machine learning
*		algorithm will be calculated. THIS VARIABLE SHOULD BE ALLOCATED
* 		AND INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* @param int n - This argument will represent the total number of samples (rows)
* 		that the input matrix has, with which the output data was
*		obtained.
*
* @param int mPlusOne - This argument will represent the total number of
*		features (independent variables) that the input matrix has plus
* 		one.
*
* @param double *TransposeOf_X_tilde - This argument will contain the pointer to
* 		a memory allocated matrix in which the transpose of the argument
* 		variable "X" in its transformed form of "X_tilde" will be stored.
* 		THIS VARIABLE SHOULD BE ALLOCATED BEFORE CALLING THIS FUNCTION
* 		WITH A SIZE OF "n" TIMES "m+1" TIMES "n" 'DOUBLE' MEMORY SPACES.
* 
*
* NOTE: RESULTS ARE STORED IN "TransposeOf_X_tilde".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 21, 2022
* LAST UPDATE: N/A
*/
__global__ static void getTransposeOfInputData_singleGPU(double *X, int n, int mPlusOne, double *TransposeOf_X_tilde) {
	// We obtain the GPU thread global coordinate.
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	// We calculate the transpose of "X_tilde" by using the argument variable "X", but only if it is within the threads boundary.
	if (idx < n) {
		double *idata = X; // We convert the pointer of interest from "X" to be the origin pointer of "idata".
		double *odata = TransposeOf_X_tilde; // We convert the pointer of interest from "TransposeOf_X_tilde" to be the origin pointer of "odata".
		odata[idx] = 1;
		for (int currentColumn=1; currentColumn<(mPlusOne); currentColumn++) {
			odata += n; // We move the origin pointer of the argument variable "TransposeOf_X_tilde" to its next column.
			odata[idx] = idata[idx]; // We apply the transpose with respect to the next column of "X_tilde".
			idata += n; // We move the origin pointer of the argument varaible "X" to its next row.
		}
	}
	
	return;
}


/**
* The "getFxTilde_Au_dAu_and_accuracyTermsPart1_singleGPU()" global static
* function is used to apply a single GPU to calculate "f(\tilde{x})", A(u),
* dA(u) and the first part of the accuracy terms calculations.
* 
* @param double *X - This argument will contain the pointer to a memory
*		allocated input matrix, from which the desired machine learning
*		algorithm will be calculated. THIS VARIABLE SHOULD BE ALLOCATED
* 		AND INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* @param double *Y - This argument will contain the pointer to a memory
* 		allocated output matrix, representing the real data of the
*		system under study. THIS VARIABLE SHOULD BE ALLOCATED AND
* 		INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
* 
* @param double *w_new - This argument will contain the pointer to a memory
*			allocated variable in which we will store the last
*			identified coefficient values for the model of a single
* 			neuron in Deep Neural Network. These coefficients will
* 			each be stored in the same row but under different
* 			columns where the first coefficient (b_0) will be stored
* 			in the column with index 0; the second coefficient (w_1)
* 			will be stored in the column index 1 and; the last
* 			coefficient (w_m) will be stored in the column index m.
* 			IT IS INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED AND
* 			INITIALIZED BEFORE CALLING THIS FUNCTION WITH A VARIABLE
* 			SIZE OF "1" TIMES "m+1" 'DOUBLE' MEMORY SPACES.
* 
* @param int n - This argument will represent the total number of samples (rows)
* 		that the input matrix has, with which the output data was
*		obtained.
*
* @param int m - This argument will represent the total number of features
* 		(independent variables) that the input matrix has, with which
* 		the output data was obtained.
*
* @param int activationFunctionToBeUsed - This argument will represent the
* 					identifier of the desired activation
* 					function to be used by the neuron during
* 					its training process. Its possible valid
* 					values are the following:
*					0 = Rectified Linear Units (ReLU).
*					1 = Hyperbolic tangent (tanh).
*					2 = Logistic function.
*					3 = Raise to the 1st power.
*					4 = Raise to the 2nd power.
*					5 = Raise to the 3rd power.
*					6 = Raise to the 4th power.
*					7 = Raise to the 5th power.
*					8 = Raise to the 6th power.
*					9 = 1st order degree exponential.
*					10 = 2nd order degree exponential.
*
* @param double *f_x_tilde - This argument will contain the pointer to a memory
* 			allocated matrix that is used to store the output of the
* 			body of the neuron in the selected GPU. IT IS
* 			INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED BEFORE
* 			CALLING THIS FUNCTION WITH A VARIABLE SIZE OF "n" TIMES
* 			"1" 'DOUBLE' MEMORY SPACES.
*
* @param double *A_u - This argument will contain the pointer to a memory
* 		allocated output matrix in which the requested activation
* 		function will be applied on the argument pointer variable
* 		"f_x_tilde" and its result will be saved in "A_u". "A_u" SHOULD
* 		BE ALLOCATED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param double *dA_u - This argument will contain the pointer to a memory
* 		allocated output matrix in which the derivate of the activation
* 		function will be applied and its result will be saved in "dA_u".
* 		"dA_u" SHOULD BE ALLOCATED BEFORE CALLING THIS FUNCTION WITH A
* 		SIZE OF "n" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param double *accuracyTerm1 - This argument will contain the pointer to a
* 			memory allocated matrix that will contain all the
* 			calculations required to obtained the SSE value (with the
* 			intention of applying the Parallel Reduction strategy to
* 			it on another process, external to this function). IT IS
* 			INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED BEFORE
* 			CALLING THIS FUNCTION WITH A VARIABLE SIZE OF "1" TIMES
* 			"n" 'DOUBLE' MEMORY SPACES.
*
* @param double *accuracyTerm2 - This argument will contain the pointer to a
* 			memory allocated matrix that will contain all the
* 			calculations required to obtained the sum of all the
* 			values contained in the argument pointer variable "Y"
* 			(with the intention of applying the Parallel Reduction
* 			strategy to it on another process, external to this
* 			function). IT IS INDISPENSABLE THAT THIS VARIABLE IS
* 			ALLOCATED BEFORE CALLING THIS FUNCTION WITH A VARIABLE
* 			SIZE OF "1" TIMES "n" 'DOUBLE' MEMORY SPACES.
* 
*
* NOTE: RESULTS ARE STORED IN "f_x_tilde", "A_u", "dA_u", "accuracyTerm1" AND
*	"accuracyTerm2".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 23, 2022
* LAST UPDATE: JANUARY 25, 2022
*/
__global__ static void getFxTilde_Au_dAu_and_accuracyTermsPart1_singleGPU(double *X, double *Y, double *w_new, int n, int m, int activationFunctionToBeUsed, double *f_x_tilde, double *A_u, double *dA_u, double *accuracyTerm1, double *accuracyTerm2) {
	// We obtain the GPU thread coordinates.
	int idx = threadIdx.x + blockIdx.x * blockDim.x; // We obtain the GPU thread global coordinate.
	int tid = threadIdx.x; // We obtain the GPU thread local coordinate
	
	// If the current GPU thread is within boundary, then proceed to work with the task. Otherwise, conclude your operation.
	if (idx < n) {
		// We calculate the values of "f(x_tilde)".
		getFxTilde(X, w_new, m, f_x_tilde, tid, idx);
		
		// We calculate the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
		getActivationFunction(activationFunctionToBeUsed, f_x_tilde, A_u, idx); // We calculate A(u) and store it in the pointer variable "A_u".
		
		// We calculate the derivative of A(u).
		// NOTE: Remember that "Y_hat" = A(u) = "A_u".
		getDerivateOfActivationFunction(activationFunctionToBeUsed, f_x_tilde, A_u, dA_u, idx); // We calculate the derivative of A(u) and store it in the pointer variable "dA_u".
		
		// We calculate the part 1 of the corresponding evaluation metric with respect to the actual data of the system under study "Y" and the currently predicted output made by the neuron "A_u".
		getNeuronAdjustedCoefficientOfDetermination_singleGPUPart1(Y, A_u, accuracyTerm1, accuracyTerm2, idx); // We calculate the part 1 of the calculation of the current adjusted coefficient of determination of the neuron.
	}
	
	return;
}
/**
* The "getFxTilde()" device static function is used to apply a single
* GPU to calculate and store the f(x_tilde) value, which stands for the output
* of the body of the single neuron in DNN.
*
*
* @param double *X - This argument will contain the pointer to a memory
*		allocated input matrix, from which the desired machine learning
*		algorithm will be calculated. THIS VARIABLE SHOULD BE ALLOCATED
* 		AND INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* @param double *Y - This argument will contain the pointer to a memory
* 		allocated output matrix, representing the real data of the
*		system under study. THIS VARIABLE SHOULD BE ALLOCATED AND
* 		INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
* 
* @param double *w_new - This argument will contain the pointer to a memory
*			allocated variable in which we will store the last
*			identified coefficient values for the model of a single
* 			neuron in Deep Neural Network. These coefficients will
* 			each be stored in the same row but under different
* 			columns where the first coefficient (b_0) will be stored
* 			in the column with index 0; the second coefficient (w_1)
* 			will be stored in the column index 1 and; the last
* 			coefficient (w_m) will be stored in the column index m.
* 			IT IS INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED AND
* 			INITIALIZED BEFORE CALLING THIS FUNCTION WITH A VARIABLE
* 			SIZE OF "1" TIMES "m+1" 'DOUBLE' MEMORY SPACES.
* 
* @param int n - This argument will represent the total number of samples (rows)
* 		that the input matrix has, with which the output data was
*		obtained.
*
* @param int m - This argument will represent the total number of features
* 		(independent variables) that the input matrix has, with which
* 		the output data was obtained.
*
* @param int activationFunctionToBeUsed - This argument will represent the
* 					identifier of the desired activation
* 					function to be used by the neuron during
* 					its training process. Its possible valid
* 					values are the following:
*					0 = Rectified Linear Units (ReLU).
*					1 = Hyperbolic tangent (tanh).
*					2 = Logistic function.
*					3 = Raise to the 1st power.
*					4 = Raise to the 2nd power.
*					5 = Raise to the 3rd power.
*					6 = Raise to the 4th power.
*					7 = Raise to the 5th power.
*					8 = Raise to the 6th power.
*					9 = 1st order degree exponential.
*					10 = 2nd order degree exponential.
*
* @param double *f_x_tilde - This argument will contain the pointer to a memory
* 			allocated matrix that is used to store the output of the
* 			body of the neuron in the selected GPU. IT IS
* 			INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED BEFORE
* 			CALLING THIS FUNCTION WITH A VARIABLE SIZE OF "n" TIMES
* 			"1" 'DOUBLE' MEMORY SPACES.
*
* @param int tid - This argument will contain the value of the GPU thread local
* 		coordinate.
*
* @param int idx - This argument will contain the value of the GPU thread global
* 		coordinate.
*
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "f_x_tilde".
* 
* @return void
* 
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 25, 2022
* LAST UPDATE: N/A
*/
__device__ static void getFxTilde(double *X, double *w_new, int m, double *f_x_tilde, int tid, int idx) {
	// We declare and initialize the shared memory of the GPU that will be used.
	extern __shared__ double sharedMem[]; // We declare the shared memory that we will use for each block.
	// NOTE: Each GPU thread is storing data such that their local address (tid) will represent the identifier of the row number in which they will write data in the shared memory. Moreover, each thread will have assigned 3 columns per row.
	
	// We calculate the values of "f(x_tilde)".
	double *idata_X = X; // We convert the pointer of interest from "X" to be the origin pointer of "idata".
	double *idata_w_new = w_new; // We convert the pointer of interest from "w_new" to be the origin pointer of "idata".
	int tidTimesThree = tid*3; // This variable is used to store a repetitive value that is used several times in the program, for performance purposes.
	int currentRowColumn1 = tidTimesThree; // This variable is used to store a repetitive value that is used several times in the program, for performance purposes.
	int currentRowColumn2 = 1 + currentRowColumn1; // This variable is used to store a repetitive value that is used several times in the program, for performance purposes.
	sharedMem[tidTimesThree] = idata_w_new[0]; // This memory address of the shared memory is where we will store the value of "f(x_tilde)". To begin with such process, we get the bias value of the body of the neuron.
	idata_w_new++; // We move the origin pointer of the argument variable "w_new" to the location of the next weight value, for performance purposes.
	idata_X += idx * m; // We move the origin pointer of the argument variable "X" to the location of the row of interest for the current GPU thread, for performance purposes.
	for (int currentColumn=0; currentColumn<m; currentColumn++) {
		currentRowColumn1++;
		currentRowColumn2++;
		sharedMem[currentRowColumn1] = idata_w_new[currentColumn]; // We pass the current weight value to the shared memory corresponding address.
		sharedMem[currentRowColumn2] = idata_X[currentColumn]; // We pass the current input data to the shared memory corresponding address.
		sharedMem[tidTimesThree] += sharedMem[currentRowColumn1] * sharedMem[currentRowColumn2]; // We multiply the current weight value with the current input data and store it in the address of the shared memory in which we will store the value of "f(x_tilde)".
	}
	f_x_tilde[idx] = sharedMem[tidTimesThree]; // We transfer the result of "f(x_tilde)" from the shared memory to the pointer variable "f_x_tilde".
	
	return;
}


/**
* The "getErrorAndUpdateWeightValues_singleGPUpart1()" global static function is
* used to apply a single GPU to make several calculations required to obtain the
* error term that is applied in the learning process of the artificial neuron
* (with the intention of applying the Parallel Reduction strategy to it on
* another process, external to this function)
* 
*
* @param double *TransposeOf_X_tilde - This argument will contain the pointer to
* 		a memory allocated matrix in which the transpose of the argument
* 		variable "X" in its transformed form of "X_tilde" will be stored.
* 		THIS VARIABLE SHOULD BE ALLOCATED AND INITIALIZED BEFORE CALLING
* 		THIS FUNCTION WITH A SIZE OF "n" TIMES "m+1" TIMES "n" 'DOUBLE'
* 		MEMORY SPACES.
*
* @param double *Y - This argument will contain the pointer to a memory
* 		allocated output matrix, representing the real data of the
*		system under study. THIS VARIABLE SHOULD BE ALLOCATED AND
* 		INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
* 
* @param int n - This argument will represent the total number of samples (rows)
* 		that the input matrix has, with which the output data was
*		obtained.
*
* @param int mPlusOne - This argument will represent the total number of
*		features (independent variables) that the input matrix has plus
* 		one.
*
* @param double *A_u - This argument will contain the pointer to a memory
* 		allocated output matrix in which the requested activation
* 		function was applied and stored. "A_u" SHOULD BE ALLOCATED AND
* 		INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param double *dA_u - This argument will contain the pointer to a memory
* 		allocated output matrix in which the derivate of the activation
* 		function was applied and stored. "dA_u" SHOULD BE ALLOCATED AND
* 		INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param double *errorTerm - This argument will contain the pointer to a
* 			memory allocated matrix that will contain all the
* 			calculations required to obtained the error term value
* 			that is applied in the learning process of the artificial
* 			neuron (with the intention of applying the Parallel
* 			Reduction strategy to it on another process, external to
* 			this function). IT IS INDISPENSABLE THAT THIS VARIABLE IS
*			ALLOCATED BEFORE CALLING THIS FUNCTION WITH A VARIABLE
* 			SIZE OF "m+1" TIMES "n" 'DOUBLE' MEMORY SPACES.
*
*
* NOTE: RESULT IS STORED IN "errorTerm".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 23, 2022
* LAST UPDATE: N/A
*/
__global__ static void getErrorAndUpdateWeightValues_singleGPUpart1(double *TransposeOf_X_tilde, double *Y, int n, int mPlusOne, double *A_u, double *dA_u, double *errorTerm) {
	// We obtain the GPU thread coordinates.
	int idx = threadIdx.x + blockIdx.x * blockDim.x; // We obtain the GPU thread global coordinate.
	
	// If the current GPU thread is within boundary, then proceed to work with the task. Otherwise, conclude your operation.
	if (idx < n) {
		// We calculate the error term contribution of the current GPU thread.
		double contributedErrorTerm = (Y[idx] - A_u[idx]) * dA_u[idx];
		
		// We calculate the contribution of the current GPU thread to obtain the dot product of the error term and the transposed matrix of X_tilde.
		double *idata1 = TransposeOf_X_tilde; // We convert the pointer of interest from "TransposeOf_X_tilde" to be the origin pointer of "idata".
		double *odata1 = errorTerm; // We convert the pointer of interest from "errorTerm" to be the origin pointer of "odata".
		for (int currentWeight=0; currentWeight<mPlusOne; currentWeight++) {
			odata1[idx] = contributedErrorTerm * idata1[idx]; // We apply the dot product between the error term and the transposed matrix of X_tilde.
			odata1 += n;
			idata1 += n;
		}
	}
	
	return;
}
/**
* The "getErrorAndUpdateWeightValues_singleGPUpart2()" global static function is
* used to employ a single GPU to apply the Parallel Reduction strategy to the
* error term contributions that should have been obtained previously (with the
* function "getErrorAndUpdateWeightValues_singleGPUpart1") to calling this
* function.
* 
*
* @param double *errorTerm - This argument will contain the pointer to a
* 			memory allocated matrix that should contain all the
* 			calculations required to obtained the error term value,
* 			which is obtained through the function
* 			"getErrorAndUpdateWeightValues_singleGPUpart1". As a
* 			result, the function
* 			"getErrorAndUpdateWeightValues_singleGPUpart2" will
* 			apply the Parallel Reduction strategy to "errorTerm",
* 			which should contain all the individual contributions to
* 			obtain the error term having already made a dot product
* 			with "X_tilde". IT IS INDISPENSABLE THAT THIS VARIABLE
* 			IS ALLOCATED AND INITIALIZED BEFORE CALLING THIS
* 			FUNCTION WITH A VARIABLE SIZE OF "m+1" TIMES "n" 'DOUBLE'
* 			MEMORY SPACES.
*
* @param int n - This argument will represent the total number of samples (rows)
* 		that the input matrix has, with which the output data was
*		obtained.
*
* @param int mPlusOne - This argument will represent the total number of
*		features (independent variables) that the input matrix has plus
* 		one.
*
* @param int trueUnrollingSize - This argument is used to represent the true
* 			number of samples that are expected to be obtained after
* 			having applied the Parallel Reduction strategy.
*
* @param int unrollingGridSize - This argument is used to represent the grid
*		size that will be considered for all the GPU Kernels that apply
* 		the Unrolling Parallel Reduction strategy, for performance
* 		purposes.
*
* @param int numberOfUnrollingLoop - This argument is used to represent the
* 				number of unrolling loops that the algorithm will
* 				use when applying the Parallel Reduction
*				Strategy.
*
* @param double *errorTerm_dot_Xtilde - This argument pointer variable is used
*				to store the data of "d_errorTerm", but after
* 				having applied the Parallel Reduction Strategy.
* 				IT IS INDISPENSABLE THAT THIS VARIABLE IS
* 				ALLOCATED BEFORE CALLING THIS FUNCTION WITH A
* 				VARIABLE SIZE OF "m+1" TIMES "GPU grid size"
* 				'DOUBLE' MEMORY SPACES.
*
*
* NOTE: RESULTS ARE STORED IN "errorTerm_dot_Xtilde".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 23, 2022
* LAST UPDATE: N/A
*/
__global__ static void getErrorAndUpdateWeightValues_singleGPUpart2(double *errorTerm, int n, int mPlusOne, int trueUnrollingSize, int unrollingGridSize, int numberOfUnrollingLoop, double *errorTerm_dot_Xtilde) {
	// We apply the parallel reduction strategy to all the individual error term contributions made for each weight available.
	double *idata2 = errorTerm; // We convert the pointer of interest from "errorTerm" to be the origin pointer of "idata".
	double *odata2 = errorTerm_dot_Xtilde; // We convert the pointer of interest from "errorTerm_dot_Xtilde" to be the origin pointer of "odata".
	for (int currentWeight=0; currentWeight<mPlusOne; currentWeight++) {
		getDeviceParallelReduction(idata2, odata2, unrollingGridSize, numberOfUnrollingLoop); // We apply the parallel reduction strategy on "errorTerm".
		idata2 += n;
		odata2 += trueUnrollingSize;
	}
	
	return;
}


/**
* The "getActivationFunction()" device static function is used to apply a single
* GPU to calculate and store the requested activation function to the output of
* the body of a neuron.
*
*
* @param int activationFunctionToBeUsed - This argument will represent the
* 					identifier of the desired activation
* 					function to be used by the neuron during
* 					its training process. Its possible valid
* 					values are the following:
*					0 = Rectified Linear Units (ReLU).
*					1 = Hyperbolic tangent (tanh).
*					2 = Logistic function.
*					3 = Raise to the 1st power.
*					4 = Raise to the 2nd power.
*					5 = Raise to the 3rd power.
*					6 = Raise to the 4th power.
*					7 = Raise to the 5th power.
*					8 = Raise to the 6th power.
*					9 = 1st order degree exponential.
*					10 = 2nd order degree exponential.
*
* @param double *u - This argument will contain the pointer to a memory
*		allocated input matrix, in which the output of the body of a
* 		neuron should be stored (f_x_tilde). THIS VARIABLE SHOULD BE
* 		ALLOCATED AND INITIALIZED BEFORE CALLING THIS FUNCTION WITH A
* 		SIZE OF "n" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param double *A_u - This argument will contain the pointer to a memory
* 		allocated output matrix in which the requested activation
* 		function was applied and stored. "A_u" SHOULD BE ALLOCATED
* 		BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n" TIMES "p=1"
* 		'DOUBLE' MEMORY SPACES.
*
* @param int idx - This argument will contain the value of the GPU thread global
* 		coordinate.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "A_u".
* 
* @return void
* 
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 21, 2022
* LAST UPDATE: N/A
*/
__device__ static void getActivationFunction(int activationFunctionToBeUsed, double *u, double *A_u, int idx) {
	// Determine and apply the activation function that was chosen by the implementer.
	// TODO: Use shared memory within the processes of this switch-case but only in the cases in which there is at least a mathematical operation to do. If you use shared memory only to transfer data like in "case 3", you will not obtain better performance results.
	double squareThisValue; // Variable used to store the value that wants to be squared, for performance purposes.
	switch (activationFunctionToBeUsed) {
		case 0: // Rectified Linear Units (ReLU).
			if (u[idx] > 0) {
				A_u[idx] = u[idx];
			} else {
				A_u[idx] = 0;
			}
			break;
		
		case 1: // Hyperbolic tangent (tanh).
			A_u[idx] = (exp(u[idx]) - exp(-u[idx])) / (exp(u[idx]) + exp(-u[idx]));
			break;
		
		case 2: // Logistic function.
			A_u[idx] = 1 / (1 + exp(-u[idx]));
			break;
		
		case 3: // Raise to the 1st power.
			A_u[idx] = u[idx];
			break;
		
		case 4: // Raise to the 2nd power.
			A_u[idx] = u[idx] * u[idx];
			break;
		
		case 5: // Raise to the 3rd power.
			A_u[idx] = u[idx] * u[idx] * u[idx];
			break;
		
		case 6: // Raise to the 4th power.
			squareThisValue = u[idx] * u[idx];
			A_u[idx] = squareThisValue * squareThisValue;
			break;
		
		case 7: // Raise to the 5th power.
			squareThisValue = u[idx] * u[idx];
			A_u[idx] = squareThisValue * squareThisValue * u[idx];
			break;
		
		case 8: // Raise to the 6th power.
			squareThisValue = u[idx] * u[idx];
			A_u[idx] = squareThisValue * squareThisValue * squareThisValue;
			break;
		
		case 9: // 1st order degree exponential.
			A_u[idx] = exp(u[idx]);
			break;
		
		default: // 2nd order degree exponential.
			A_u[idx] = exp(u[idx] * u[idx]);
	}
	return;
}


/**
* The "getDerivateOfActivationFunction()" device static function is
* used to apply a single GPU to calculate and store the derivate of the
* requested activation function that should have been applied to the output of
* the body of a neuron.
*
*
* @param int activationFunctionToBeUsed - This argument will represent the
* 					identifier of the desired activation
* 					function to be used by the neuron during
* 					its training process. Its possible valid
* 					values are the following:
*					0 = Rectified Linear Units (ReLU).
*					1 = Hyperbolic tangent (tanh).
*					2 = Logistic function.
*					3 = Raise to the 1st power.
*					4 = Raise to the 2nd power.
*					5 = Raise to the 3rd power.
*					6 = Raise to the 4th power.
*					7 = Raise to the 5th power.
*					8 = Raise to the 6th power.
*					9 = 1st order degree exponential.
*					10 = 2nd order degree exponential.
*
* @param double *u - This argument will contain the pointer to a memory
*		allocated input matrix, in which the output of the body of a
* 		neuron should be stored (f_x_tilde). THIS VARIABLE SHOULD BE
* 		ALLOCATED AND INITIALIZED BEFORE CALLING THIS FUNCTION WITH A
* 		SIZE OF "n" TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param double *A_u - This argument will contain the pointer to a memory
* 		allocated output matrix in which the requested activation
* 		function was applied and stored. "A_u" SHOULD BE ALLOCATED AND
* 		INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param double *dA_u - This argument will contain the pointer to a memory
* 		allocated output matrix in which the derivate of the activation
* 		function was applied and stored. "dA_u" SHOULD BE ALLOCATED
* 		BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n" TIMES "p=1"
* 		'DOUBLE' MEMORY SPACES.
*
* @param int idx - This argument will contain the value of the GPU thread global
* 		coordinate.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "dA_u".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 21, 2022
* LAST UPDATE: N/A
*/
__device__ static void getDerivateOfActivationFunction(int activationFunctionToBeUsed, double *u, double *A_u, double *dA_u, int idx) {
	// Determine and apply the derivate of the activation function that was chosen by the implementer.
	// TODO: Use shared memory within the processes of this switch-case but only in the cases in which there is at least a mathematical operation to do. If you use shared memory only to transfer data like in "case 3", you will not obtain better performance results.
	double squareThisValue; // Variable used to store the value that wants to be squared, for performance purposes.
	switch (activationFunctionToBeUsed) {
		case 0: // Rectified Linear Units (ReLU).
			if (u[idx] > 0) {
				dA_u[idx] = 1;
			} else {
				dA_u[idx] = 0;
			}
			break;
		
		case 1: // Hyperbolic tangent (tanh).
			dA_u[idx] = 1 - A_u[idx] * A_u[idx];
			break;
		
		case 2: // Logistic function.
			dA_u[idx] = A_u[idx] * (1 - A_u[idx]);
			break;
		
		case 3: // Raise to the 1st power.
			dA_u[idx] = 1;
			break;
		
		case 4: // Raise to the 2nd power.
			dA_u[idx] = 2*u[idx];
			break;
		
		case 5: // Raise to the 3rd power.
			dA_u[idx] = 3 * u[idx] * u[idx];
			break;
		
		case 6: // Raise to the 4th power.
			dA_u[idx] = 4 * u[idx] * u[idx] * u[idx];
			break;
		
		case 7: // Raise to the 5th power.
			squareThisValue = u[idx] * u[idx];
			dA_u[idx] = 5 * squareThisValue * squareThisValue;
			break;
		
		case 8: // Raise to the 6th power.
			squareThisValue = u[idx] * u[idx];
			dA_u[idx] = 6 * squareThisValue * squareThisValue * u[idx];
			break;
		
		case 9: // 1st order degree exponential.
			dA_u[idx] = A_u[idx];
			break;
		
		default: // 2nd order degree exponential.
			dA_u[idx] = 2 * u[idx] * A_u[idx];
	}
	return;
}


/**
* The "getNeuronAdjustedCoefficientOfDetermination_singleGPUPart1()" device
* static function is used to apply a single GPU to make several calculations
* required to obtain the adjusted R-squared evaluation metric with respect to
* the actual data of the system under study "Y" and the currently predicted
* output made by the neuron "A_u" (with the intention of applying the Parallel
* Reduction strategy to it on another process, external to this function).
* 
*
* @param double *Y - This argument will contain the pointer to a memory
* 		allocated output matrix, representing the real data of the
*		system under study. THIS VARIABLE SHOULD BE ALLOCATED AND
* 		INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param double *A_u - This argument will contain the pointer to a memory
* 		allocated output matrix in which the requested activation
* 		function was applied and stored. "A_u" SHOULD BE ALLOCATED AND
* 		INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
*
* @param double *accuracyTerm1 - This argument will contain the pointer to a
* 			memory allocated matrix that will contain all the
* 			calculations required to obtained the SSE value (with the
* 			intention of applying the Parallel Reduction strategy to
* 			it on another process, external to this function). IT IS
* 			INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED BEFORE
* 			CALLING THIS FUNCTION WITH A VARIABLE SIZE OF "1" TIMES
* 			"n" 'DOUBLE' MEMORY SPACES.
*
* @param double *accuracyTerm2 - This argument will contain the pointer to a
* 			memory allocated matrix that will contain all the
* 			calculations required to obtained the sum of all the
* 			values contained in the argument pointer variable "Y"
* 			(with the intention of applying the Parallel Reduction
* 			strategy to it on another process, external to this
* 			function). IT IS INDISPENSABLE THAT THIS VARIABLE IS
* 			ALLOCATED BEFORE CALLING THIS FUNCTION WITH A VARIABLE
* 			SIZE OF "1" TIMES "n" 'DOUBLE' MEMORY SPACES.
*
* @param int idx - This argument will contain the value of the GPU thread global
* 		coordinate.
*
*
* NOTE: RESULTS ARE STORED IN "accuracyTerm1" and "accuracyTerm2".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 24, 2022
* LAST UPDATE: JANUARY 25, 2022
*/
__device__ static void getNeuronAdjustedCoefficientOfDetermination_singleGPUPart1(double *Y, double *A_u, double *accuracyTerm1, double *accuracyTerm2, int idx) {
	// We obtain the GPU thread local coordinate.
	int tid = threadIdx.x;
	
	// We declare and initialize the shared memory of the GPU that will be used.
	extern __shared__ double sharedMem[]; // We declare the shared memory that we will use for each block.
	// NOTE: Each GPU thread is storing data such that their local address (tid) will represent the identifier of the row number in which they will write data in the shared memory. Moreover, each thread will have assigned 2 columns per row.
	int column1 = tid * 2;
	int column2 = 1 + column1;
	sharedMem[column1] = Y[idx];
	sharedMem[column2] = A_u[idx];
	
	// We obtain and store all the GPU threads contibutions to calculate the sum of the real output matrix.
	accuracyTerm2[idx] = sharedMem[column1]; // We temporarily store the sum of the real output matrix in the argument pointer variable "accuracyTerm2", for performance purposes.
	
	// We obtain and store all the GPU threads contibutions to calculate the SSE value.
	sharedMem[column1] = sharedMem[column1] - sharedMem[column2]; // real output matrix - predicted output matrix
	sharedMem[column2] = sharedMem[column1] * sharedMem[column1]; // We square the value that was previously obtained.
	accuracyTerm1[idx] = sharedMem[column2]; // We temporarly store the SSE values in the argument pointer variable "accuracyTerm1", for performance purposes.
	
	return;
}


/**
* The "getParallelReduction()" global static function is used to employ a single
* GPU to apply the Parallel Reduction strategy to the argument variable
* "termToBeReduced".
* 
*
* @param double *termToBeReduced - This argument will contain the pointer to a
* 				memory allocated variable on which it is desired
* 				to apply the Parallel Reduction strategy.
*
* @param double *reducedAccuracyTerm - This argument will contain the pointer to
* 				a memory allocated variable in which we will
* 				store the result of having applied the Parallel
* 				Reduction strategy to the argument pointer
* 				variable "termToBeReduced".
*
* @param int n - This argument will represent the total number of samples (rows)
* 		that the input matrix has, with which the output data was
*		obtained.
*
* @param int numberOfUnrollingLoop - This argument is used to represent the
* 				number of unrolling loops that the algorithm will
* 				use when applying the Parallel Reduction
*				Strategy.
*
* NOTE: RESULTS ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLES
*       "reducedAccuracyTerm".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 24, 2022
* LAST UPDATE: N/A
*/
__global__ static void getParallelReduction(double *termToBeReduced, double *reducedAccuracyTerm, int n, int numberOfUnrollingLoop) {
	// We declare the variables that will be given a value through the next case code.
	int idx; // Variable used to store the GPU thread global coordinate.
	int tid = threadIdx.x; // Variable used to store the GPU thread local coordinate.
	double *idata; // Variable used to convert the pointer of interest from "termToBeReduced" to be the origin pointer of "idata".
	double unroll1; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll2; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll3; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll4; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll5; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll6; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll7; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll8; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll9; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll10; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	
	// Parallel Reduction Strategy: Unrolling Strategy process.
	switch (numberOfUnrollingLoop) {
		case 10: // "Unrolling10 Strategy": Unrolling 10 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 10*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 9*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx]; idata += blockDim.x;
				unroll6 = idata[idx]; idata += blockDim.x;
				unroll7 = idata[idx]; idata += blockDim.x;
				unroll8 = idata[idx]; idata += blockDim.x;
				unroll9 = idata[idx]; idata += blockDim.x;
				unroll10 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5 + unroll6 + unroll7 + unroll8 + unroll9 + unroll10;
			}
			idata = termToBeReduced + 10*blockIdx.x*blockDim.x;
		break;
		
		case 9: // "Unrolling9 Strategy": Unrolling 9 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 9*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 8*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx]; idata += blockDim.x;
				unroll6 = idata[idx]; idata += blockDim.x;
				unroll7 = idata[idx]; idata += blockDim.x;
				unroll8 = idata[idx]; idata += blockDim.x;
				unroll9 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5 + unroll6 + unroll7 + unroll8 + unroll9;
			}
			idata = termToBeReduced + 9*blockIdx.x*blockDim.x;
		break;
		
		case 8: // "Unrolling8 Strategy": Unrolling 8 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 8*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 7*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx]; idata += blockDim.x;
				unroll6 = idata[idx]; idata += blockDim.x;
				unroll7 = idata[idx]; idata += blockDim.x;
				unroll8 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5 + unroll6 + unroll7 + unroll8;
			}
			idata = termToBeReduced + 8*blockIdx.x*blockDim.x;
		break;
		
		case 7: // "Unrolling7 Strategy": Unrolling 7 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 7*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 6*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx]; idata += blockDim.x;
				unroll6 = idata[idx]; idata += blockDim.x;
				unroll7 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5 + unroll6 + unroll7;
			}
			idata = termToBeReduced + 7*blockIdx.x*blockDim.x;
		break;
		
		case 6: // "Unrolling6 Strategy": Unrolling 6 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 6*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 5*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx]; idata += blockDim.x;
				unroll6 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5 + unroll6;
			}
			idata = termToBeReduced + 6*blockIdx.x*blockDim.x;
		break;
		
		case 5: // "Unrolling5 Strategy": Unrolling 5 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 5*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 4*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5;
			}
			idata = termToBeReduced + 5*blockIdx.x*blockDim.x;
		break;
		
		case 4: // "Unrolling4 Strategy": Unrolling 4 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 4*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 3*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4;
			}
			idata = termToBeReduced + 4*blockIdx.x*blockDim.x;
		break;
		
		case 3: // "Unrolling3 Strategy": Unrolling 3 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 3*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 2*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3;
			}
			idata = termToBeReduced + 3*blockIdx.x*blockDim.x;
		break;
		
		case 2: // "Unrolling2 Strategy": Unrolling 2 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 2*blockIdx.x*blockDim.x;
			idata = termToBeReduced + 2*blockIdx.x*blockDim.x;
			if ((idx + blockDim.x) < n) {
				unroll1 = termToBeReduced[idx];
				unroll2 = termToBeReduced[idx + blockDim.x];
				termToBeReduced[idx] = unroll1 + unroll2;
			}
		break;
		
		default: // No "Unrolling Strategy" will be applied.
			idx = threadIdx.x + blockIdx.x*blockDim.x;
			idata = termToBeReduced + blockIdx.x*blockDim.x;
	}
	__syncthreads(); // We synchronize all threads within the same block.
	
	// Parallel Reduction Strategy: Unrolling Warp process with shared memory.
	extern __shared__ double sharedMem[]; // We declare the shared memory that we will use for each block.
	sharedMem[tid] = idata[tid];
	__syncthreads(); // We synchronize all threads within the same block.
	if (tid < 16) {
		sharedMem[tid] += sharedMem[tid+16];__syncthreads(); // We synchronize all threads within the same block.
		sharedMem[tid] += sharedMem[tid+8];__syncthreads(); // We synchronize all threads within the same block.
		sharedMem[tid] += sharedMem[tid+4];__syncthreads(); // We synchronize all threads within the same block.
		sharedMem[tid] += sharedMem[tid+2];__syncthreads(); // We synchronize all threads within the same block.
		sharedMem[tid] += sharedMem[tid+1];__syncthreads(); // We synchronize all threads within the same block.
	}
	
	// Store the final result corresponding to the current block.
	if (tid == 0) reducedAccuracyTerm[blockIdx.x] = sharedMem[0];
	
	return;
}


/**
* The "getDeviceParallelReduction()" device static function is used to employ a
* single GPU to apply the Parallel Reduction strategy to the argument variable
* "termToBeReduced".
*
*
* @param double *termToBeReduced - This argument will contain the pointer to a
* 				memory allocated variable on which it is desired
* 				to apply the Parallel Reduction strategy.
*
* @param double *reducedAccuracyTerm - This argument will contain the pointer to
* 				a memory allocated variable in which we will
* 				store the result of having applied the Parallel
* 				Reduction strategy to the argument pointer
* 				variable "termToBeReduced".
*
* @param int n - This argument will represent the total number of samples (rows)
* 		that the input matrix has, with which the output data was
*		obtained.
*
* @param int numberOfUnrollingLoop - This argument is used to represent the
* 				number of unrolling loops that the algorithm will
* 				use when applying the Parallel Reduction
*				Strategy.
*
* NOTE: RESULTS ARE STORED IN THE MEMORY ALLOCATED POINTER VARIABLES
*       "reducedAccuracyTerm".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 24, 2022
* LAST UPDATE: N/A
*/
__device__ static void getDeviceParallelReduction(double *termToBeReduced, double *reducedAccuracyTerm, int n, int numberOfUnrollingLoop) {
	// We declare the variables that will be given a value through the next case code.
	int idx; // Variable used to store the GPU thread global coordinate.
	int tid = threadIdx.x; // Variable used to store the GPU thread local coordinate.
	double *idata; // Variable used to convert the pointer of interest from "termToBeReduced" to be the origin pointer of "idata".
	double unroll1; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll2; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll3; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll4; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll5; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll6; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll7; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll8; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll9; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	double unroll10; // Variable used in the "Unrolling Loop Strategy" that applies, if any.
	
	// Parallel Reduction Strategy: Unrolling Strategy process.
	switch (numberOfUnrollingLoop) {
		case 10: // "Unrolling10 Strategy": Unrolling 10 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 10*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 9*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx]; idata += blockDim.x;
				unroll6 = idata[idx]; idata += blockDim.x;
				unroll7 = idata[idx]; idata += blockDim.x;
				unroll8 = idata[idx]; idata += blockDim.x;
				unroll9 = idata[idx]; idata += blockDim.x;
				unroll10 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5 + unroll6 + unroll7 + unroll8 + unroll9 + unroll10;
			}
			idata = termToBeReduced + 10*blockIdx.x*blockDim.x;
		break;
		
		case 9: // "Unrolling9 Strategy": Unrolling 9 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 9*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 8*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx]; idata += blockDim.x;
				unroll6 = idata[idx]; idata += blockDim.x;
				unroll7 = idata[idx]; idata += blockDim.x;
				unroll8 = idata[idx]; idata += blockDim.x;
				unroll9 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5 + unroll6 + unroll7 + unroll8 + unroll9;
			}
			idata = termToBeReduced + 9*blockIdx.x*blockDim.x;
		break;
		
		case 8: // "Unrolling8 Strategy": Unrolling 8 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 8*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 7*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx]; idata += blockDim.x;
				unroll6 = idata[idx]; idata += blockDim.x;
				unroll7 = idata[idx]; idata += blockDim.x;
				unroll8 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5 + unroll6 + unroll7 + unroll8;
			}
			idata = termToBeReduced + 8*blockIdx.x*blockDim.x;
		break;
		
		case 7: // "Unrolling7 Strategy": Unrolling 7 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 7*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 6*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx]; idata += blockDim.x;
				unroll6 = idata[idx]; idata += blockDim.x;
				unroll7 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5 + unroll6 + unroll7;
			}
			idata = termToBeReduced + 7*blockIdx.x*blockDim.x;
		break;
		
		case 6: // "Unrolling6 Strategy": Unrolling 6 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 6*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 5*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx]; idata += blockDim.x;
				unroll6 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5 + unroll6;
			}
			idata = termToBeReduced + 6*blockIdx.x*blockDim.x;
		break;
		
		case 5: // "Unrolling5 Strategy": Unrolling 5 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 5*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 4*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx]; idata += blockDim.x;
				unroll5 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4 + unroll5;
			}
			idata = termToBeReduced + 5*blockIdx.x*blockDim.x;
		break;
		
		case 4: // "Unrolling4 Strategy": Unrolling 4 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 4*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 3*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx]; idata += blockDim.x;
				unroll4 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3 + unroll4;
			}
			idata = termToBeReduced + 4*blockIdx.x*blockDim.x;
		break;
		
		case 3: // "Unrolling3 Strategy": Unrolling 3 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 3*blockIdx.x*blockDim.x;
			idata = termToBeReduced;
			if ((idx + 2*blockDim.x) < n) {
				unroll1 = idata[idx]; idata += blockDim.x;
				unroll2 = idata[idx]; idata += blockDim.x;
				unroll3 = idata[idx];
				termToBeReduced[idx] = unroll1 + unroll2 + unroll3;
			}
			idata = termToBeReduced + 3*blockIdx.x*blockDim.x;
		break;
		
		case 2: // "Unrolling2 Strategy": Unrolling 2 times, but only with the GPU threads that are within the boundary.
			idx = threadIdx.x + 2*blockIdx.x*blockDim.x;
			idata = termToBeReduced + 2*blockIdx.x*blockDim.x;
			if ((idx + blockDim.x) < n) {
				unroll1 = termToBeReduced[idx];
				unroll2 = termToBeReduced[idx + blockDim.x];
				termToBeReduced[idx] = unroll1 + unroll2;
			}
		break;
		
		default: // No "Unrolling Strategy" will be applied.
			idx = threadIdx.x + blockIdx.x*blockDim.x;
			idata = termToBeReduced + blockIdx.x*blockDim.x;
	}
	__syncthreads(); // We synchronize all threads within the same block.
	
	// Parallel Reduction Strategy: Unrolling Warp process with shared memory.
	extern __shared__ double sharedMem[]; // We declare the shared memory that we will use for each block.
	sharedMem[tid] = idata[tid];
	__syncthreads(); // We synchronize all threads within the same block.
	if (tid < 16) {
		sharedMem[tid] += sharedMem[tid+16];__syncthreads(); // We synchronize all threads within the same block.
		sharedMem[tid] += sharedMem[tid+8];__syncthreads(); // We synchronize all threads within the same block.
		sharedMem[tid] += sharedMem[tid+4];__syncthreads(); // We synchronize all threads within the same block.
		sharedMem[tid] += sharedMem[tid+2];__syncthreads(); // We synchronize all threads within the same block.
		sharedMem[tid] += sharedMem[tid+1];__syncthreads(); // We synchronize all threads within the same block.
	}
	
	// Store the final result corresponding to the current block.
	if (tid == 0) reducedAccuracyTerm[blockIdx.x] = sharedMem[0];
	
	return;
}


/**
* The "getNeuronAdjustedCoefficientOfDetermination_singleGPUvoidPart2()" global static
* function is used to apply a single GPU to calculate "f(\tilde{x})", A(u),
* dA(u) and the first part of the accuracy terms calculations.
* 
* @param double *Y - This argument will contain the pointer to a memory
* 		allocated output matrix, representing the real data of the
*		system under study. THIS VARIABLE SHOULD BE ALLOCATED AND
* 		INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
* 
* @param int n - This argument will represent the total number of samples (rows)
* 		that the input matrix has, with which the output data was
*		obtained.
*
* @param double *accuracyTerm1 - This argument will contain the pointer to a
* 			memory allocated matrix that will contain all the
* 			calculations required to obtained the SSE value (with the
* 			intention of applying the Parallel Reduction strategy to
* 			it on another process, external to this function). IT IS
* 			INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED BEFORE
* 			CALLING THIS FUNCTION WITH A VARIABLE SIZE OF "1" TIMES
* 			"n" 'DOUBLE' MEMORY SPACES.
*
* @param double *reducedAccuracyTerm2 - This argument will contain the pointer
* 				to a memory allocated matrix that will contain
* 				the mean of the real output matrix ("Y") in the
* 				address with the identifier "0" in it. IT IS
* 				INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED AND
* 				INITIALIZED BEFORE CALLING THIS FUNCTION WITH A
* 				VARIABLE SIZE OF "1" TIMES "grid size of the GPU
* 				Kernel" 'DOUBLE' MEMORY SPACES.
* 
*
* NOTE: RESULTS ARE STORED IN "accuracyTerm1".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 24, 2022
* LAST UPDATE: JANUARY 25, 2022
*/
__global__ static void getNeuronAdjustedCoefficientOfDetermination_singleGPUvoidPart2(double *Y, int n, double *accuracyTerm1, double *reducedAccuracyTerm2) {
	// We obtain the GPU threads coordinates.
	int tid = threadIdx.x; // We obtain the GPU thread local coordinate
	int idx = threadIdx.x + blockIdx.x * blockDim.x; // We obtain the GPU thread global coordinate.
	
	// If the current GPU thread is within boundary, then proceed to work with the task. Otherwise, conclude your operation.
	if (idx < n) {
		// We declare and initialize the shared memory of the GPU that will be used.
		extern __shared__ double sharedMem[]; // We declare the shared memory that we will use for each block.
		// NOTE: Each GPU thread is storing data such that their local address (tid) will represent the identifier of the row number in which they will write data in the shared memory. Moreover, each thread will have assigned 2 columns per row.
		int column1 = tid * 2;
		int column2 = 1 + column1;
		sharedMem[column1] = Y[idx];
		sharedMem[column2] = reducedAccuracyTerm2[0];
		
		// We get the MSSE value.
		sharedMem[column1] = sharedMem[column1] - sharedMem[column2];
		sharedMem[column2] = sharedMem[column1] * sharedMem[column1];
		accuracyTerm1[idx] = sharedMem[column2];
	}
	
	return;
}


/**
* The "predictSingleNeuronDNN_singleGPU()" function is used to apply a single
* GPU to make the predictions of the requested input values (X) by applying the
* simple linear equation system with the specified coefficient values (b). The
* predicted values will be stored in the argument pointer variable "Y_hat".
* 
* @param struct singleNeuronDnnStruct_singleGPU *neuron - This argument will
* 					contain the pointer to a struct variable
* 					that should contain all the information
* 					required in order to be able to create
* 					and make an artificial neuron. Its
* 					accessible inner elements are described
* 					in the list showed in the commented
* 					documentation of the function
* 					"getSingleNeuronDNN_singleGPU()".
*
* @param double *Y_hat - This argument will contain the pointer to a memory
* 			allocated output matrix, representing the predicted data
* 			of the system under study. THIS VARIABLE SHOULD BE
* 			ALLOCATED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 			TIMES "p=1" 'DOUBLE' MEMORY SPACES. The results will be
* 			stored in the same order as the input data given such
* 			that the first sample will be stored in the row with
* 			index "0" and the last sample in the row with index "n".
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE "Y_hat".
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 24, 2022
* LAST UPDATE: JANUARY 25, 2022
*/
void predictSingleNeuronDNN_singleGPU(struct singleNeuronDnnStruct_singleGPU *neuron, double *Y_hat) {
	// If the requested GPU (device) is less than zero, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (neuron->gpuDevice < 0) {
		printf("\nERROR: The identifier of the requested GPU (device) must be equal or greater than 0.\n");
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
	// If the output of the system under study is different than the value of one, then emit an error message and terminate the program. Otherwise, continue with the program.
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
	
	
	// ------- SELECTION AND INITIALIZATION OF THE DESIRED GPU ------- //
	// We selected the GPU desired by the implementer and inform in the terminal the name of such GPU.
	cudaDeviceProp gpuProperties;
	CHECK(cudaGetDeviceProperties(&gpuProperties, neuron->gpuDevice)); // We obtain the details of the GPU that was defined by the implementer.
	printf("\nThe GPU (device) %d: %s, has been selected by the CenyML library.\n", neuron->gpuDevice, gpuProperties.name);
	CHECK(cudaSetDevice(neuron->gpuDevice)); // We select the GPU that was requested by the implementer.
	
	// Set up the execution configurations that will be assigned to the selected GPU.
	dim3 block_32x_1y(32, 1); // We define the number of GPU threads per block.
	dim3 grid_n((neuron->n + block_32x_1y.x - 1) / block_32x_1y.x, 1); // We define the number of blocks that our GPU will manage.
	
	// We configure the shared memory of the current GPU.
	cudaSharedMemConfig pConfig = cudaSharedMemBankSizeEightByte; // We create a cudaSharedMemConfig type variable to store in it the configuration of 8-byte mode for shared memory in the GPU.
	cudaDeviceSetSharedMemConfig(pConfig); // We set the 8-byte mode for shared memory in the selected GPU.
	
	
	// --------------- PREPROCESSING OF THE INPUT DATA --------------- //
	// We create the pointers to the data that the selected GPU will require.
	double *d_X; // This pointer variable is used to store the data from "neuron->X" into the selected GPU.
	double *d_w_new; // This pointer variable is used to store the data from "neuron->w_new" into the selected GPU.
	double *d_f_x_tilde; // This pointer variable is used to store the output of the body of the neuron in the selected GPU.
	double *d_A_u; // This pointer variable is used to store the output of the application of the chosen activation function in the selected GPU.
	
	// We allocate the required memory in the selected GPU.
	int nDoubles_Bytes = neuron->n*sizeof(double); // This variable stores the number of bytes to allocate "n" bytes of double type, for performance purposes.
	CHECK(cudaMalloc((void **) &d_X, neuron->m*nDoubles_Bytes));
	int w_new_Bytes = (neuron->m+1)*sizeof(double); // This variable stores the number of bytes to allocate the new weight values that will be obtained each epoch of the training process, for performance purposes.
	CHECK(cudaMalloc((void **) &d_w_new, w_new_Bytes));
	CHECK(cudaMalloc((void **) &d_f_x_tilde, nDoubles_Bytes));
	CHECK(cudaMalloc((void **) &d_A_u, nDoubles_Bytes));
	
	// We transfer the required data from the CPU to the selected GPU.
	CHECK(cudaMemcpy(d_w_new, neuron->w_best, w_new_Bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_X, neuron->X, (neuron->n*neuron->m*sizeof(double)), cudaMemcpyHostToDevice));
	
	
	// --------------- DATA PREDICTION PROCESS --------------- //
	// We obtain the requested predictions from the artificial neuron model.
	getPredictSingleNeuronDNN_singleGPU <<< grid_n, block_32x_1y, (3*32*sizeof(double)) >>> (d_X, d_w_new, neuron->n, neuron->m, neuron->activationFunctionToBeUsed, neuron->isClassification, neuron->threshold, neuron->desiredValueForGroup1, neuron->desiredValueForGroup2, d_f_x_tilde, d_A_u);
	
	// We transfer the predicted data from the GPU to the CPU, to the argument variable "Y_hat".
	CHECK(cudaMemcpy(Y_hat, d_A_u, nDoubles_Bytes, cudaMemcpyDeviceToHost));
	
	
	// Before terminating this function, we free the GPU and CPU allocated memory since they will no longer be used.
	CHECK(cudaFree(d_X));
	CHECK(cudaFree(d_w_new));
	CHECK(cudaFree(d_f_x_tilde));
	CHECK(cudaFree(d_A_u));
	return;
}


/**
* The "getPredictSingleNeuronDNN_singleGPU()" global static
* function is used to apply a single GPU to calculate the prediction made by a specified single
* artificial nueron model.
*
* @param double *X - This argument will contain the pointer to a memory
*		allocated input matrix, from which the desired machine learning
*		algorithm will be calculated. THIS VARIABLE SHOULD BE ALLOCATED
* 		AND INITIALIZED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "m" 'DOUBLE' MEMORY SPACES.
*
* @param double *w_new - This argument will contain the pointer to a memory
*			allocated variable in which we will store the last
*			identified coefficient values for the model of a single
* 			neuron in Deep Neural Network. These coefficients will
* 			each be stored in the same row but under different
* 			columns where the first coefficient (b_0) will be stored
* 			in the column with index 0; the second coefficient (w_1)
* 			will be stored in the column index 1 and; the last
* 			coefficient (w_m) will be stored in the column index m.
* 			IT IS INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED AND
* 			INITIALIZED BEFORE CALLING THIS FUNCTION WITH A VARIABLE
* 			SIZE OF "1" TIMES "m+1" 'DOUBLE' MEMORY SPACES.
* 
* @param int n - This argument will represent the total number of samples (rows)
* 		that the input matrix has, with which the output data was
*		obtained.
*
* @param int m - This argument will represent the total number of features
* 		(independent variables) that the input matrix has, with which
* 		the output data was obtained.
*
* @param int activationFunctionToBeUsed - This argument will represent the
* 					identifier of the desired activation
* 					function to be used by the neuron during
* 					its training process. Its possible valid
* 					values are the following:
*					0 = Rectified Linear Units (ReLU).
*					1 = Hyperbolic tangent (tanh).
*					2 = Logistic function.
*					3 = Raise to the 1st power.
*					4 = Raise to the 2nd power.
*					5 = Raise to the 3rd power.
*					6 = Raise to the 4th power.
*					7 = Raise to the 5th power.
*					8 = Raise to the 6th power.
*					9 = 1st order degree exponential.
*					10 = 2nd order degree exponential.
*
* @param char isClassification = This argument variable will work as a flag to
* 				indicate to the neron if it is expected from it
* 				to interpret the given data of "X" and "Y" as if
* 				their were meant for a classification problem or
* 				not. The possible valid values for this flag are
* 				the following:
*				1) "isClassification" = (int) 1 --> The neuron
* 				will interpret the data of "X" and "Y" as if they
* 				were meant for a classification problem.
*				2) "isClassification" = (int) 0 --> The neuron
* 				will interpret the data of "X" and "Y" as if they
* 				were meant for a regression problem.
*
* @param double threshold - This argument will represent desired threshold that
* 			the implementer desired the neuron to consider in
* 			classification problems. In this regard, whenever the
* 			predicted output of the neuron is higher than the
* 			defined threshold value, then that prediction should be
*			interpreted as group 1 (ussually refered to as the binary
* 			output 1). Conversely, if the predicted value is lower
* 			than the defined threshold value, then that prediction
* 			should be interpreted as group 2 (ussually refered to as
*			the binary output 0). However, have in mind that
* 			"threshold" will only be used by the neuron if the
* 			argument variable "isClassification" = 1.
*
* @param int desiredValueForGroup1 - This argument will represent the desired
*				label value to whenever an output of the neuron
* 				predicts the classification group 1. Ussually,
* 				this is label with the value of "(int) 1" but any
* 				other customized value can be assigned by the
* 				implementer. However, have in mind that this
* 				argument variable will be considered by the
* 				neuron as long as the argument variable
*				"isClassification" = 1 and only when the
*				implementer requests to the neuron a prediction
* 				through the function
* 				"predictSingleNeuronDNN_singleGPU()".
*
* @param int desiredValueForGroup2 - This argument will represent the desired
*				label value to whenever an output of the neuron
* 				predicts the classification group 2. Ussually,
* 				this is label with the value of "(int) -1" but
* 				any other customized value can be assigned by the
* 				implementer. However, have in mind that this
* 				argument variable will be considered by the
* 				neuron as long as the argument variable
*				"isClassification" = 1 and only when the
*				implementer requests to the neuron a prediction
* 				through the function
* 				"predictSingleNeuronDNN_singleGPU()".
*
* @param double *f_x_tilde - This argument will contain the pointer to a memory
* 			allocated matrix that is used to store the output of the
* 			body of the neuron in the selected GPU. IT IS
* 			INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED BEFORE
* 			CALLING THIS FUNCTION WITH A VARIABLE SIZE OF "n" TIMES
* 			"1" 'DOUBLE' MEMORY SPACES.
*
* @param double *A_u - This argument will contain the pointer to a memory
* 		allocated output matrix in which the requested activation
* 		function will be applied on the argument pointer variable
* 		"f_x_tilde" and its result will be saved in "A_u". "A_u" SHOULD
* 		BE ALLOCATED BEFORE CALLING THIS FUNCTION WITH A SIZE OF "n"
* 		TIMES "p=1" 'DOUBLE' MEMORY SPACES.
* 
*
* NOTE: RESULTS ARE STORED IN "A_u".
*
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 24, 2022
* LAST UPDATE: JANUARY 25, 2022
*/
__global__ static void getPredictSingleNeuronDNN_singleGPU(double *X, double *w_new, int n, int m, int activationFunctionToBeUsed, int isClassification, double threshold, int desiredValueForGroup1, int desiredValueForGroup2, double *f_x_tilde, double *A_u) {
	// We obtain the GPU thread coordinates.
	int idx = threadIdx.x + blockIdx.x * blockDim.x; // We obtain the GPU thread global coordinate.
	int tid = threadIdx.x; // We obtain the GPU thread local coordinate
	
	// If the current GPU thread is within boundary, then proceed to work with the task. Otherwise, conclude your operation.
	if (idx < n) {
		// We calculate the values of "f(x_tilde)".
		getFxTilde(X, w_new, m, f_x_tilde, tid, idx);
		
		// We calculate the currently predicted output data made by the neuron and store it in "A_u" by applying the desired activation function to "f_x_tilde".
		getActivationFunction(activationFunctionToBeUsed, f_x_tilde, A_u, idx); // We calculate A(u) and store it in the pointer variable "A_u".
		
		// Determine if the given model of a single neuron in Deep Neural Network is meant for a classification or for a regression problem to then make the predictions accordingly.
		if (isClassification == 1) {
			// We apply the threshold defined by the implementer in order to obtain a classification output and store it in "A_u".
			if (A_u[idx] > threshold) {
				A_u[idx] = desiredValueForGroup1; // Group 1 has been predicted.
			} else {
				A_u[idx] = desiredValueForGroup2; // Group 2 has been predicted.
			}
		}
	}
	
	return;
}

