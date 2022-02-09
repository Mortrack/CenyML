/*
* This program will read a .csv file containing the data of a linear equation
* system to then extract all its data. Its input data will be saved into the
* matrix "X" and its output data into the matrix "Y". Subsequently, a single
* neuron in Deep Neural Network will be used to obtain the best fitting
* coefficient values of such data by applying such algorithm with a multi GPU.
* Then, some evaluation metrics will be applied. Next, two new .csv files will
* be created to save: 1) the coefficient values that were obtained and 2) the
* results obtained with the evaluation metrics. Finally, a plot of the predicted
* data by the obtained model with respect to the actual data, will be plotted
* and saved into a .png file. Both the .csv files and this .png file will serve
* for further comparisons and validation purposes.
*/

 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
#include "../../../../CenyML_library_skeleton/otherLibraries/time/mTime.h" // library to count the time elapsed in Linux Ubuntu.
#include "../../../../CenyML_library_skeleton/otherLibraries/csv/csvManager.h" // library to open and create .csv files.
#include "../../../../CenyML_library_skeleton/otherLibraries/pbPlots/pbPlots.h" // library to generate plots v0.1.9.0
#include "../../../../CenyML_library_skeleton/otherLibraries/pbPlots/supportLib.h"  // library required for "pbPlots.h" v0.1.9.0
#include "../../../../CenyML_library_skeleton/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.h" // library to use the classification evaluation metrics of CenyML.
#include "../../../../CenyML_library_skeleton/CenyML_Library/multiGpu/machineLearning/CenyMLdeepLearning_MG.h" // library to use the deep learning algorithms of CenyML with multi GPU parallelism.
#include "../../../../CenyML_library_skeleton/otherLibraries/cuda/CUDA_check.h" // library to use a define function to allow the tracking of CUDA errors in the terminal window.
#include <cuda_runtime.h>


// ---------------------------------------------- //
// ----- DEFINE THE GLOBAL RESOURCES TO USE ----- //
// ---------------------------------------------- //


// --------------------------------------------------- //
// ----- The code from the main file starts here ----- //
// --------------------------------------------------- //



// ----------------------------------------------- //
// ----- DEFINE THE GENERAL FUNCTIONS TO USE ----- //
// ----------------------------------------------- //
/**
* The "linspace()" function is inspired in the code made by
* "SamuraiMelon" in the stackoverflow web site under the title of
* "Linearly Spaced Array in C" (URL = https://bit.ly/3r5Dom6).
* Nonetheless, as coded in this file, this function is used to
* allocate some memory space whose pointer variable will be used
* to create a linearly spaced array with respect to the specified
* values in the argument variables of this function.
*
* @param double startFrom - This argument will be used as the
*							starting value of the linearly spaced
*							array to be created.
*
* @param double endHere - This argument will be used as the ending
*						  value of the linearly spaced array to be
*						  created.
*
* @param int n - This argument will represent the total number of
*				 rows that the linearly spaced array to be created
*				 will have.
*
*
* @return double *vector - Pointer Variable with "n" rows and "1"
*						   columns that will be used as a linearly
*						   spaced array with "n" spaces and that
*						   will start with the value specified in
*						   the argument variable "startFrom" and
*						   that will end with the value specified
*						   in the argument variable "endHere".
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 17, 2021
* LAST UPDATE: N/A
*/
double *linspace(double startFrom, double endHere, int n) {
	// We allocate the required memory to create the pointer variable "vector" in which we will make a linearly spaced array.
	double *vector;
	double step = (endHere - startFrom) / (double) n;
	vector = (double *) malloc(n*sizeof(double));
	
	// We store the values of the pointer variable "vector".
	vector[0] = startFrom; // We store the initial value of the pointer variable "vector".
	for (int currentRow=1; currentRow<n; currentRow++) { // We store the values of the pointer variable "vector" from row index "1" up to row index "n-1".
	 vector[currentRow] = vector[currentRow - 1] + step;
	}
	vector[n-1] = endHere; // We store the last value of the pointer variable "vector".
	
	// We return the address of the allocated variable "vector".
	return vector;
}


// ----------------------------------------- //
// ----- THE MAIN FUNCTION STARTS HERE ----- //
// ----------------------------------------- //
/**
* This is the main function of the program. Here we will read a .csv file and
* then apply the single neuron in Deep Neural Network on the input and output
* data contained in it by applying such algorithm with a multi GPU. In
* addition, some evaluation metrics will be applied to evaluate the model.
* Finally, the results will be saved in two new .csv files and in a .png file
* for further comparison and validation purposes.
*
* @param int argc - This argument will possess the length number of what is
*		    contained within the argument "*argv[]".
*		    NOTE1: This argument will be at least "1" in value because
*		    its first argument is the title of this program.
*
* @param char **argv[] - This double pointer argument contains within the following:
*		       ARGUMENT1 = The directory of this program, including its name.
*		       ARGUMENT2 = All the characters you input on the terminal.
*
* @return 0
*
* @author Miranda Meza Cesar
* CREATION DATE: JANUARY 17, 2021
* LAST UPDATE: N/A
*/
int main(int argc, char **argv) {
	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	char csv1Directory[] = "../../../../databases/classification/linearEquationSystem/100systems_100samplesPerAxisPerSys.csv"; // Directory of the reference .csv file
	char nameOfTheCsvFile1[] = "CenyML_getLinearClassification_Coefficients.csv"; // Name the .csv file that will store the resulting coefficient values.
	char nameOfTheCsvFile2[] = "CenyML_getLinearClassification_evalMetrics.csv"; // Name the .csv file that will store the resulting evaluation metrics for the ML model to be obtained.
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file (which is declared in "csvManager.h").
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	int columnIndexOfOutputDataInCsvFile = 2; // This variable will contain the index of the first column in which we will specify the location of the real output values (Y).
	int columnIndexOfInputDataInCsvFile = 3; // This variable will contain the index of the first column in which we will specify the location of the input values (X).
	struct singleNeuronDnnStruct_multiGPU neuron1; // We create a singleNeuronDnnStruct_multiGPU structure variable to manage the data input and output data of the single neuron in DNN that will be created.
	neuron1.firstGpuDevice = 1; // This variable will define the identifier of the first identifier of the GPU device that wants to be used to parallize the algorithms of the single neuron in DNN with multiple GPU.
	neuron1.lastGpuDevice = 4; // This variable will define the identifier of the last identifier of the GPU device that wants to be used to parallize the algorithms of the single neuron in DNN with multiple GPU.
	neuron1.maxUnrollingLoop = 15; // This variable will define the desired maximum Unrolling Loop strategy that wants to be applied within the Parallel Reduction and Unrolling Warp strategies (these are applied in the single neuron in DNN algorithm).
	neuron1.m = 2; // This variable will contain the number of features (independent variables) that the input matrix is expected to have.
	neuron1.p = 1; // This variable will contain the number of outputs that the output matrix is expected to have.
	neuron1.isInitial_w = 1; // This variable will indicate whether or not initial values will be given by the implementer (with value of 1) or if random ones are going to be used (with value of 0).
	neuron1.w_first = (double *) malloc((neuron1.m+1)*sizeof(double)); // We allocate the memory required for the variable "neuron1.w_first", which will store the initial coefficient values of the neuron to be created.
	neuron1.w_first[0] = 0; // We define the customized desired value for the bias of the neuron to be created.
	neuron1.w_first[1] = 0; // We define the customized desired value for the weight_1 value of the neuron to be created.
	neuron1.w_first[2] = 0; // We define the customized desired value for the weight_2 value of the neuron to be created.
	neuron1.isClassification = 1; // This variable will indicate whether or not it is desired that the neuron considers the input data for a classification (with a vlaue of 1) or a regression problem (with a value of 0).
	neuron1.threshold = 0; // This variable will be used to store the desired threshold value to be used in classification problems by the neuron to be created.
	neuron1.desiredValueForGroup1 = 1; // This variable will be used to store the label to be used for the group 1 in classification problems by the neuron to be created.
	neuron1.desiredValueForGroup2 = -1; // This variable will be used to store the label to be used for the group 2 in classification problems by the neuron to be created.
	neuron1.activationFunctionToBeUsed = 3; // This variable tells the neuron what activation function to use (see the commented documentation in the function "getSingleNeuronDNN()" for more details).
	neuron1.learningRate = 0.0000000001; // This variable stores the desired learning rate for the neuron to be created.
	neuron1.stopAboveThisAccuracy = 0.60; // The value of this variable is used as a stop function for the single neuron in DNN learning process.
	neuron1.maxEpochs = 50000; // This variable stores the desired value for the maximum permitted epochs for the training process of the neuron1.
	neuron1.isReportLearningProgress = 1; // The value of this variable tells the neuron if it is desired that it reports its learning progress (with a value of 1) or not (with a value of 0).
	neuron1.reportEachSpecifiedEpochs = neuron1.maxEpochs / 10; // This variable tells the neuron that it has to report each several times, which is defined by the value contained in this variable.
	
	
	// ---------------------- IMPORT DATA TO USE --------------------- //
	printf("Initializing data extraction from .csv file containing the data to be used ...\n");
	double startingTime, elapsedTime; // Declaration of variables used to count time in seconds.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to obtain the data from the reference .csv file.
	// Obtain the rows and columns dimensions of the data of the csv file (excluding headers)
	csv1.rowsAndColumnsDimensions = (int *) malloc(2*sizeof(int)); // We initialize the variable that will store the rows & columns dimensions.
	getCsvRowsAndColumnsDimensions(&csv1); // We input the memory location of the "csv1" into the argument of this function to get the rows & columns dimensions.
	// We save the rows and columns dimensions obtained in some variables that relate to the mathematical symbology according to the documentation of the method to be validated.
	neuron1.n = csv1.rowsAndColumnsDimensions[0]; // total number of rows of the input matrix (neuron1.X)
	int databaseColumns1 = csv1.rowsAndColumnsDimensions[1]; // total number of columns of the database that was opened.
	// From the structure variable "csv1", we allocate the memory required for the variable (csv1.allData) so that we can store the data of the .csv file in it.
	csv1.allData = (double *) malloc(neuron1.n*databaseColumns1*sizeof(double));
	// We retrieve the data contained in the reference .csv file.
	getCsvFileData(&csv1); // We input the memory location of the "csv1" into the argument of this function to get all the data contained in the .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the data from the reference .csv file.
	printf("Data extraction from .csv file containing %d samples for each of the %d columns (total samples = %d), elapsed %f seconds.\n\n", neuron1.n, databaseColumns1, (neuron1.n*databaseColumns1), elapsedTime);
	
	
	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	printf("Initializing the output and input data with %d samples for each of the %d columns (total samples = %d) each...\n", neuron1.n, neuron1.m, (neuron1.n*neuron1.m));
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to innitialize the input data to be used.
	// Allocate the memory required for the variable "neuron1.Y", which will contain the real output data of the system under study.
	CHECK(cudaMallocHost((void **) &neuron1.Y, neuron1.n*neuron1.p*sizeof(double))); // We allocate the required CPU memory with page locked memory for asynchronous data transfer.
	double *original_Y = (double *) malloc(neuron1.n*neuron1.p*sizeof(double));
	// Allocate the memory required for the variable "neuron1.X", which will contain the input data of the system under study.
	CHECK(cudaMallocHost((void **) &neuron1.X, neuron1.n*neuron1.m*sizeof(double))); // We allocate the required CPU memory with page locked memory for asynchronous data transfer.
	// Create the output (neuron1.Y) and input (neuron1.X) data with the same rows as in the reference .csv file and their corresponding number of columns.
	for (int currentRow=0; currentRow<neuron1.n; currentRow++) { // Since neuron1.m=neuron1.p=1 for this particular ML algorithm, both "neuron1.Y" and "neuron1.X" will be innitialized here.
		original_Y[currentRow] = csv1.allData[columnIndexOfOutputDataInCsvFile + currentRow*databaseColumns1];
		if (original_Y[currentRow] == 0) {
			neuron1.Y[currentRow] = -1;
		} else {
			neuron1.Y[currentRow] = 1;
		}
		for (int currentColumn=0; currentColumn<neuron1.m; currentColumn++) {
			neuron1.X[currentColumn + currentRow*neuron1.m] = csv1.allData[columnIndexOfInputDataInCsvFile + currentColumn + currentRow*databaseColumns1];
		}
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to innitialize the input data to be used.
	printf("Output and input data innitialization elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------------------- DATA MODELING ----------------------- //
	printf("Initializing CenyML single neuron in Deep Neural Network algorithm ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to apply the single neuron in Deep Neural Network with the input data (neuron1.X).
	neuron1.w_best = (double *) malloc((neuron1.m+1)*sizeof(double)); // We allocate the memory required for the variable "neuron1.w_best", which will store the best coefficient values identified by the neuron to be created, after its training process.
	CHECK(cudaMallocHost((void **) &neuron1.w_new, (neuron1.m+1)*sizeof(double))); // We allocate the required CPU memory with page locked memory for asynchronous data transfer.
	// We apply the single neuron in Deep Neural Network algorithm with respect to the input matrix "neuron1.X" and the result is stored in the memory location of the pointer "b".
	getSingleNeuronDNN_multiGPU(&neuron1);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to apply the single neuron in Deep Neural Network with the input data (neuron1.X).
	printf("CenyML single neuron in Deep Neural Network algorithm elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	// We predict the input values (neuron1.X) with the machine learning model that was obtained.
	printf("Initializing CenyML predictions with the model that was obtained ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to apply the prediction with the model that was obtained.
	// Allocate the memory required for the variable "Y_hat", which will contain the predicted output data of the system under study.
	double *Y_hat;
	CHECK(cudaMallocHost((void **) &Y_hat, neuron1.n*sizeof(double))); // We allocate the required CPU memory with page locked memory for asynchronous data transfer.
	// We obtain the predicted values with the machine learning model that was obtained.
	predictSingleNeuronDNN_multiGPU(&neuron1, Y_hat);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the prediction wit hthe model that was obtained.
	printf("The CenyML predictions with the model that was obtained elapsed %f seconds.\n\n", elapsedTime);
	
	// We change the values of "-1" for "0" so that we can use the predicted values in the evaluation metric funtions.
	for (int currentRow=0; currentRow<neuron1.n; currentRow++) {
		if (Y_hat[currentRow] == -1) {
			Y_hat[currentRow] = 0;
		}
	}
	
	// We apply the cross entropy error metric.
	double NLLepsilon = 1.0E-15; // This variable will contain the user desired epsilon value to be summed to any zero value and substracted to any value of the output matrixes (Y and/or Y_hat). NOTE: It will be assigned the value to match the one used in scikit-learn.
	printf("Initializing CenyML cross entropy error metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the cross entropy error metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "NLL" (which will contain the results of the cross entropy error metric between "Y" and "Y_hat").
	double *NLL = (double *) calloc(1, sizeof(double));
	// We apply the cross entropy error metric between "Y" and "Y_hat".
	getCrossEntropyError(original_Y, Y_hat, neuron1.n, NLLepsilon, NLL);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the cross entropy error metric between "Y" and "Y_hat".
	printf("CenyML cross entropy error metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the confusion matrix metric.
	printf("Initializing CenyML confusion matrix metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the confusion matrix metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "confusionMatrix" (which will contain the results of the confusion matrix metric between "Y" and "Y_hat").
	double *confusionMatrix = (double *) calloc(4, sizeof(double));
	// We apply the confusion matrix metric between "Y" and "Y_hat".
	getConfusionMatrix(original_Y, Y_hat, neuron1.n, confusionMatrix);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the confusion matrix metric between "Y" and "Y_hat".
	printf("CenyML confusion matrix metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the accuracy metric.
	printf("Initializing CenyML accuracy metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the accuracy metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "accuracy" (which will contain the results of the accuracy metric between "Y" and "Y_hat").
	double *accuracy = (double *) calloc(1, sizeof(double));
	// We apply the accuracy metric between "Y" and "Y_hat".
	getAccuracy(original_Y, Y_hat, neuron1.n, accuracy);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the accuracy metric between "Y" and "Y_hat".
	printf("CenyML accuracy metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the precision metric.
	printf("Initializing CenyML precision metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the precision metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "precision" (which will contain the results of the precision metric between "Y" and "Y_hat").
	double *precision = (double *) calloc(1, sizeof(double));
	// We apply the precision metric between "Y" and "Y_hat".
	getPrecision(original_Y, Y_hat, neuron1.n, precision);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the precision metric between "Y" and "Y_hat".
	printf("CenyML precision metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the recall metric.
	printf("Initializing CenyML recall metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the recall metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "recall" (which will contain the results of the recall metric between "Y" and "Y_hat").
	double *recall = (double *) calloc(1, sizeof(double));
	// We apply the recall metric between "Y" and "Y_hat".
	getRecall(original_Y, Y_hat, neuron1.n, recall);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the recall metric between "Y" and "Y_hat".
	printf("CenyML recall metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the F1 score metric.
	printf("Initializing CenyML F1 score metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the F1 score metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "F1 score" (which will contain the results of the F1 score metric between "Y" and "Y_hat").
	double *F1score = (double *) calloc(1, sizeof(double));
	// We apply the F1 score metric between "Y" and "Y_hat".
	getF1score(original_Y, Y_hat, neuron1.n, F1score);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the F1 score metric between "Y" and "Y_hat".
	printf("CenyML F1 score metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We create a single variable that contains within all the evaluation metrics that were tested.
	printf("Initializing single variable that will store all the evaluation metrics done ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the initialization of the single variable that will store all the evaluation metrics.
	// Allocate the memory required for the variable "evaluationMetrics" (which will contain all the results of the evaluation metrics that were obtained).
	double *evaluationMetrics = (double *) malloc(9*sizeof(double));
	evaluationMetrics[0] = NLL[0]; // We add the cross entropy error metric.
	evaluationMetrics[1] = confusionMatrix[0]; // We add the true positives from the confusion matrix.
	evaluationMetrics[2] = confusionMatrix[1]; // We add the false positives from the confusion matrix.
	evaluationMetrics[3] = confusionMatrix[2]; // We add the false negatives from the confusion matrix.
	evaluationMetrics[4] = confusionMatrix[3]; // We add the true negatives from the confusion matrix.
	evaluationMetrics[5] = accuracy[0]; // We add the accuracy metric.
	evaluationMetrics[6] = precision[0]; // We add the precision metric.
	evaluationMetrics[7] = recall[0]; // We add the recall metric.
	evaluationMetrics[8] = F1score[0]; // We add the F1 score metric.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the initialization of the single variable that will store all the evaluation metrics.
	printf("Innitialization of single variable to store all the evaluation metrics elapsed %f seconds.\n\n", elapsedTime);
	
	// We store the coefficients that were obtained.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .csv file which will store the coefficients that were obtained.
	// Define the desired header names for the new .csv file to be create.
    char csvHeaders1[strlen("coefficients")+1]; // Variable where the following code will store the .csv headers.
    csvHeaders1[0] = '\0'; // Innitialize this char variable with a null value.
	strcat(csvHeaders1, "coefficients"); // We add the headers into "csvHeaders".
	// Create a new .csv file and save the results obtained in it.
	char is_nArray1 = 0; // Indicate through this flag variable that the variable that indicates the samples (n) is not an array because it has the same amount of samples per columns.
	char isInsertId1 = 0; // Indicate through this flag variable that it is not desired that the file to be created automatically adds an "id" to each row.
	int csvFile_n1 = neuron1.m+1; // This variable is used to indicate the number of rows with data that will be printed in the .csv file to be created.
	createCsvFile(nameOfTheCsvFile1, csvHeaders1, neuron1.w_best, &csvFile_n1, is_nArray1, neuron1.p, isInsertId1); // We create the desired .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Creation of the .csv file to store the coefficients that were obtained, elapsed %f seconds.\n\n", elapsedTime);
	
	// We store the resulting evaluation metrics that were obtained.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .csv file which will store the results of the evaluation metrics that were obtained.
	// Define the desired header names for the new .csv file to be create.
    char csvHeaders2[strlen("NLL, TP, FP, FN, TN, accuracy, precision, recall, F1score")+1]; // Variable where the following code will store the .csv headers.
    csvHeaders2[0] = '\0'; // Innitialize this char variable with a null value.
	strcat(csvHeaders2, "NLL, TP, FP, FN, TN, accuracy, precision, recall, F1score"); // We add the headers into "csvHeaders".
	// Create a new .csv file and save the results obtained in it.
	char is_nArray2 = 0; // Indicate through this flag variable that the variable that indicates the samples (1) is not an array because it has the same amount of samples per columns.
	char isInsertId2 = 0; // Indicate through this flag variable that it is not desired that the file to be created automatically adds an "id" to each row.
	int csvFile_n2 = 1; // This variable is used to indicate the number of rows with data that will be printed in the .csv file to be created.
	createCsvFile(nameOfTheCsvFile2, csvHeaders2, evaluationMetrics, &csvFile_n2, is_nArray2, 9, isInsertId2); // We create the desired .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Creation of the .csv file to store the evaluation metrics that were obtained, elapsed %f seconds.\n\n", elapsedTime);
	
	
	
	// Plot a graph with the model that was obtained and saved it into a .png file.
	printf("Initializing creation of .png image to store the plot of the predicted data and the actual data ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .png file that will store the results of the predicted and actual data.
	// Trying the "pbPlots" library (https://github.com/InductiveComputerScience/pbPlots)
	_Bool success;
    StringReference *errorMessage;
	RGBABitmapImageReference *imageReference = CreateRGBABitmapImageReference();
	
	// In order to continue with the plotting process, identify the minimum and maximum values contained in each machine learning feature.
	double minX1 = neuron1.X[0];
	double maxX1 = neuron1.X[0];
	double minX2 = neuron1.X[1];
	double maxX2 = neuron1.X[1];
	for (int currentRow=1; currentRow<neuron1.n; currentRow++) {
		if (neuron1.X[currentRow*neuron1.m] < minX1) {
			minX1 = neuron1.X[currentRow*neuron1.m];
		}
		if (neuron1.X[currentRow*neuron1.m] > maxX1) {
			maxX1 = neuron1.X[currentRow*neuron1.m];
		}
		if (neuron1.X[1 + currentRow*neuron1.m] < minX2) {
			minX2 = neuron1.X[1 + currentRow*neuron1.m];
		}
		if (neuron1.X[1 + currentRow*neuron1.m] > maxX2) {
			maxX2 = neuron1.X[1 + currentRow*neuron1.m];
		}
	}
	// In order to continue with the plotting process, we increase by a 50% the ranges of the minimum and maximum detected.
	double rangeToBeAdded; // This variable will be used to store the range to be added/decreased for each max and min value detected.
	rangeToBeAdded = (maxX1 - minX1) * 0.5;
	minX1 = minX1 - rangeToBeAdded;
	maxX1 = maxX1 + rangeToBeAdded;
	rangeToBeAdded = (maxX2 - minX2) * 0.5;
	minX2 = minX2 - rangeToBeAdded;
	maxX2 = maxX2 + rangeToBeAdded;
	// In order to continue with the plotting process, we create some linearly spaced vectors of each independent feature with the min and max ranges that were obtained.
	int n_ofLinearlySpacedArray = 100;
	double *X1 = linspace(minX1, maxX1, n_ofLinearlySpacedArray);
	double *X2 = linspace(minX2, maxX2, n_ofLinearlySpacedArray);
	// In order to continue with the plotting process, we create a new input matrix that contains the data of the vectors that were created, which will be used to create the background of the plot to be created.
	struct singleNeuronDnnStruct_multiGPU bgNeuron; // We create a singleNeuronDnnStruct_singleGPU structure variable to manage the data input and output data of the single neuron in DNN that will be created.
	bgNeuron.firstGpuDevice = neuron1.firstGpuDevice; // This variable will contain the number of CPU threads that wants to be used to parallelize the training and predictions made by the neuron to be created.
	bgNeuron.lastGpuDevice = neuron1.lastGpuDevice; // This variable will contain the number of CPU threads that wants to be used to parallelize the training and predictions made by the neuron to be created.
	bgNeuron.maxUnrollingLoop = neuron1.maxUnrollingLoop; // This variable will contain the number of CPU threads that wants to be used to parallelize the training and predictions made by the neuron to be created.
	bgNeuron.m = neuron1.m; // This variable will contain the number of features (independent variables) that the input matrix is expected to have.
	bgNeuron.p = neuron1.p; // This variable will contain the number of outputs that the output matrix is expected to have.
	bgNeuron.w_best = (double *) malloc((neuron1.m+1)*sizeof(double)); // We allocate the memory required for the variable "bgNeuron.w_first", which will store the initial coefficient values of the neuron to be created.
	bgNeuron.w_best[0] = neuron1.w_best[0]; // We define the customized desired value for the bias of the neuron to be created.
	bgNeuron.w_best[1] = neuron1.w_best[1]; // We define the customized desired value for the weight_1 value of the neuron to be created.
	bgNeuron.w_best[2] = neuron1.w_best[2]; // We define the customized desired value for the weight_2 value of the neuron to be created.
	bgNeuron.isClassification = neuron1.isClassification; // This variable will indicate whether or not it is desired that the neuron considers the input data for a classification (with a vlaue of 1) or a regression problem (with a value of 0).
	bgNeuron.threshold = neuron1.threshold; // This variable will be used to store the desired threshold value to be used in classification problems by the neuron to be created.
	bgNeuron.desiredValueForGroup1 = neuron1.desiredValueForGroup1; // This variable will be used to store the label to be used for the group 1 in classification problems by the neuron to be created.
	bgNeuron.desiredValueForGroup2 = neuron1.desiredValueForGroup2; // This variable will be used to store the label to be used for the group 2 in classification problems by the neuron to be created.
	bgNeuron.activationFunctionToBeUsed = neuron1.activationFunctionToBeUsed; // This variable tells the neuron what activation function to use (see the commented documentation in the function "getSingleNeuronDNN()" for more details).
	bgNeuron.X = (double *) malloc((n_ofLinearlySpacedArray*n_ofLinearlySpacedArray)*neuron1.m*sizeof(double)); // Allocate the memory required for the variable "bg_X".
	int currentRow_bg_X=0;
	for (int currentRow1=0; currentRow1<n_ofLinearlySpacedArray; currentRow1++) { // Store the data that must be contained in the input matrix "bg_X".
		for (int currentRow2=0; currentRow2<n_ofLinearlySpacedArray; currentRow2++) {
			bgNeuron.X[0 + currentRow_bg_X*neuron1.m] = X1[currentRow1];
			bgNeuron.X[1 + currentRow_bg_X*neuron1.m] = X2[currentRow2];
			currentRow_bg_X++;
		}
	}
	// In order to continue the plotting process, we obtain the output data that will be used to create the background of the plot to be created.
	double *bg_Y_hat = (double *) malloc((n_ofLinearlySpacedArray*n_ofLinearlySpacedArray)*neuron1.p*sizeof(double));
	bgNeuron.n = n_ofLinearlySpacedArray*n_ofLinearlySpacedArray;
	predictSingleNeuronDNN_multiGPU(&bgNeuron, bg_Y_hat);
	// In order to continue the plotting process, we determine the number of 1s and 0s that were predicted for the background to be created.
	int n_of_bg1s = 0; // This variable will be used as a counter to determine the rows length of the pointer variables "bg_X1_1s" and "bg_X2_1s" to be created.
	int n_of_bg0s = 0; // This variable will be used as a counter to determine the rows length of the pointer variables "bg_X1_0s" and "bg_X2_0s" to be created.
	for (int currentRow=0; currentRow<(n_ofLinearlySpacedArray*n_ofLinearlySpacedArray); currentRow++) {
		if (bg_Y_hat[currentRow] == 1) {
			n_of_bg1s++;
		} else {
			n_of_bg0s++;
		}
	}
	// In order to continue the plotting process, we seperate the input data that has an output value of "1" with respect to the ones that have an output of "0".
	double *bg_X1_1s = (double *) malloc(n_of_bg1s*1*sizeof(double)); // Allocate the memory required for the variable "bg_X1_1s".
	double *bg_X2_1s = (double *) malloc(n_of_bg1s*1*sizeof(double)); // Allocate the memory required for the variable "bg_X2_1s".
	double *bg_X1_0s = (double *) malloc(n_of_bg0s*1*sizeof(double)); // Allocate the memory required for the variable "bg_X1_0s".
	double *bg_X2_0s = (double *) malloc(n_of_bg0s*1*sizeof(double)); // Allocate the memory required for the variable "bg_X2_0s".
	int currentRow_bg_X_1s = 0; // This variable will be used as a counter for the output values of "1" that the pointer variables "bg_X1_1s" and "bg_X2_1s" have.
	int currentRow_bg_X_0s = 0; // This variable will be used as a counter for the output values of "0" that the pointer variables "bg_X1_0s" and "bg_X2_0s" have.
	for (int currentRow=0; currentRow<(n_ofLinearlySpacedArray*n_ofLinearlySpacedArray); currentRow++) {
		if (bg_Y_hat[currentRow] == 1) {
			bg_X1_1s[currentRow_bg_X_1s] = bgNeuron.X[currentRow*neuron1.m];
			bg_X2_1s[currentRow_bg_X_1s] = bgNeuron.X[1 + currentRow*neuron1.m];
			currentRow_bg_X_1s++;
		} else {
			bg_X1_0s[currentRow_bg_X_0s] = bgNeuron.X[currentRow*neuron1.m];
			bg_X2_0s[currentRow_bg_X_0s] = bgNeuron.X[1 + currentRow*neuron1.m];
			currentRow_bg_X_0s++;
		}
	}
	
	// background with 1s
	ScatterPlotSeries *series = GetDefaultScatterPlotSeriesSettings();
	series->xs = bg_X1_1s;
	series->xsLength = n_of_bg1s;
	series->ys = bg_X2_1s;
	series->ysLength = n_of_bg1s;
	series->linearInterpolation = false;
	series->pointType = L"dots";
	series->pointTypeLength = wcslen(series->pointType);
	series->color = CreateRGBAColor(0.808, 0.922, 0.804, 0.05);
	
	// background with 0s
	
	ScatterPlotSeries *series2 = GetDefaultScatterPlotSeriesSettings();
	series2->xs = bg_X1_0s;
	series2->xsLength = n_of_bg0s;
	series2->ys = bg_X2_0s;
	series2->ysLength = n_of_bg0s;
	series2->linearInterpolation = false;
	series2->pointType = L"dots";
	series2->pointTypeLength = wcslen(series2->pointType);
	series2->color = CreateRGBAColor(0.902, 0.749, 0.749, 0.05);
	
	// In order to continue the plotting process, we determine the number of 1s and 0s that were predicted by the machine learning model that was created.
	int n_of_Y1s = 0; // This variable will be used as a counter to determine the rows length of the pointer variables "real_X1_1s" and "real_X2_1s" to be created.
	int n_of_Y0s = 0; // This variable will be used as a counter to determine the rows length of the pointer variables "real_X1_0s" and "real_X2_0s" to be created.
	for (int currentRow=0; currentRow<neuron1.n; currentRow++) {
		if (original_Y[currentRow] == 1) {
			n_of_Y1s++;
		} else {
			n_of_Y0s++;
		}
	}
	// In order to continue the plotting process, we seperate the input data that has an output value of "1" with respect to the ones that have an output of "0".
	double *real_X1_1s = (double *) malloc(n_of_Y1s*sizeof(double)); // Allocate the memory required for the variable "real_X1_1s".
	double *real_X2_1s = (double *) malloc(n_of_Y1s*sizeof(double)); // Allocate the memory required for the variable "real_X2_1s".
	double *real_X1_0s = (double *) malloc(n_of_Y0s*sizeof(double)); // Allocate the memory required for the variable "real_X1_0s".
	double *real_X2_0s = (double *) malloc(n_of_Y0s*sizeof(double)); // Allocate the memory required for the variable "real_X2_0s".
	int currentRow_predicted_X_1s = 0; // This variable will be used as a counter for the output values of "1" that the pointer variables "real_X1_1s" and "real_X2_1s" have.
	int currentRow_predicted_X_0s = 0; // This variable will be used as a counter for the output values of "0" that the pointer variables "real_X1_0s" and "real_X2_0s" have.
	for (int currentRow=0; currentRow<neuron1.n; currentRow++) {
		if (original_Y[currentRow] == 1) {
			real_X1_1s[currentRow_predicted_X_1s] = neuron1.X[currentRow*neuron1.m];
			real_X2_1s[currentRow_predicted_X_1s] = neuron1.X[1 + currentRow*neuron1.m];
			currentRow_predicted_X_1s++;
		} else {
			real_X1_0s[currentRow_predicted_X_0s] = neuron1.X[currentRow*neuron1.m];
			real_X2_0s[currentRow_predicted_X_0s] = neuron1.X[1 + currentRow*neuron1.m];
			currentRow_predicted_X_0s++;
		}
	}
	
	// real data with 1s
	ScatterPlotSeries *series3 = GetDefaultScatterPlotSeriesSettings();
	series3->xs = real_X1_1s;
	series3->xsLength = n_of_Y1s;
	series3->ys = real_X2_1s;
	series3->ysLength = n_of_Y1s;
	series3->linearInterpolation = false;
	series3->pointType = L"dots";
	series3->pointTypeLength = wcslen(series3->pointType);
	series3->color = CreateRGBColor(0.325, 0.890, 0);
	// real data with 0s
	ScatterPlotSeries *series4 = GetDefaultScatterPlotSeriesSettings();
	series4->xs = real_X1_0s;
	series4->xsLength = n_of_Y0s;
	series4->ys = real_X2_0s;
	series4->ysLength = n_of_Y0s;
	series4->linearInterpolation = false;
	series4->pointType = L"filled triangles";
	series4->pointTypeLength = wcslen(series4->pointType);
	series4->color = CreateRGBColor(0.929, 0.196, 0.216);
	
	// This next series will be created because, due to a bug, the pbPlots library assings automatically one last point at the end.
	// Therefore, we will plot a "filled triangles" scatter point on top of it with color black so that it does not make the user think of that point as an actual data of interest.
	double xs [] = {0};
	double ys [] = {0};
	ScatterPlotSeries *series5 = GetDefaultScatterPlotSeriesSettings();
	series5->xs = xs;
	series5->xsLength = 1;
	series5->ys = ys;
	series5->ysLength = 1;
	series5->linearInterpolation = false;
	series5->pointType = L"filled triangles";
	series5->pointTypeLength = wcslen(series5->pointType);
	series5->color = GetBlack();
	
	// Create the .png image with the desired plot.
	ScatterPlotSettings *settings = GetDefaultScatterPlotSettings();
	settings->width = 600;
	settings->height = 400;
	settings->autoBoundaries = true;
	settings->autoPadding = true;
	settings->title = L"";
	settings->titleLength = wcslen(settings->title);
	settings->xLabel = L"";
	settings->xLabelLength = wcslen(settings->xLabel);
	settings->yLabel = L"";
	settings->yLabelLength = wcslen(settings->yLabel);
	ScatterPlotSeries *s [] = {series2, series, series3, series4, series5};
	settings->scatterPlotSeries = s;
	settings->scatterPlotSeriesLength = 5;
	
	// If there is an error during the .png file creation, show it on the terminal window.
    errorMessage = (StringReference *)malloc(sizeof(StringReference));
	success = DrawScatterPlotFromSettings(imageReference, settings, errorMessage);
    if(success){
        size_t length;
        double *pngdata = ConvertToPNG(&length, imageReference->image);
        WriteToFile(pngdata, length, "plotOfMachineLearningModel (CenyML).png");
        DeleteImage(imageReference->image);
	}else{
	    fprintf(stderr, "Error: ");
        for(int i = 0; i < errorMessage->stringLength; i++){
            fprintf(stderr, "%c", errorMessage->string[i]);
        }
        fprintf(stderr, "\n");
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .png file that will store the results of the predicted and actual data.
	printf("Initialization of the creation of the .png file elapsed %f seconds.\n", elapsedTime);
	printf("NOTE: In regards to the .png image, the horizontal axis stands for the independent variable 1 and the vertical axis for the independent variable 2.\n");
	printf("In addition, the color green stands for an output value of 1 and the color red for an output value of -1.\n");
	printf("Finally, while the background color represents the predicted 1s and -1s that were made by the machine learning model that was just trained, the dots/triangles stand for the true behaviour of the system under study.\n\n");
	printf("The program has been successfully completed!\n");
	
	// Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
	free(neuron1.w_first);
	free(csv1.rowsAndColumnsDimensions);
	free(csv1.allData);
	CHECK(cudaFreeHost(neuron1.Y));
	CHECK(cudaFreeHost(neuron1.X));
	free(original_Y);
	free(neuron1.w_best);
	CHECK(cudaFreeHost(neuron1.w_new));
	CHECK(cudaFreeHost(Y_hat));
	free(NLL);
	free(confusionMatrix);
	free(accuracy);
	free(precision);
	free(recall);
	free(F1score);
	free(evaluationMetrics);
	free(X1);
	free(X2);
	free(bgNeuron.X);
	free(bgNeuron.w_best);
	free(bg_Y_hat);
	free(bg_X1_1s);
	free(bg_X2_1s);
	free(bg_X1_0s);
	free(bg_X2_0s);
	free(real_X1_1s);
	free(real_X2_1s);
	free(real_X1_0s);
	free(real_X2_0s);
	free(errorMessage);
	return (0); // end of program.
}

