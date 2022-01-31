/*
* This program will read a .csv file containing the data of a linear
* equation system to then extract all its data. Its input data will be
* saved into the matrix "X" and then an output data (matrix "Y") will
* be generated through this program. Subsequently, a single neuron in
* Deep Neural Network will be used to obtain the best fitting
* coefficient values of such data by applying such algorithm with CPU
* parallelism through POSIX Threads. Then, some evaluation metrics will
* be applied. Next, two new .csv files will be created to save: 1) the
* coefficient values that were obtained and 2) the results obtained
* with the evaluation metrics. Finally, a plot of the predicted data
* by the obtained model with respect to the actual data, will be
* plotted and saved into a .png file. Both the .csv files and this
* .png file will serve for further comparisons and validation purposes.
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
#include "../../../../CenyML_library_skeleton/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.h" // library to use the regression evaluation metrics of CenyML.
#include "../../../../CenyML_library_skeleton/CenyML_Library/cpuParallel/machineLearning/CenyMLdeepLearning_PC.h" // library to use the deep learning algorithms of CenyML with CPU parallelism.


// ---------------------------------------------- //
// ----- DEFINE THE GLOBAL RESOURCES TO USE ----- //
// ---------------------------------------------- //


// --------------------------------------------------- //
// ----- The code from the main file starts here ----- //
// --------------------------------------------------- //



// ----------------------------------------------- //
// ----- DEFINE THE GENERAL FUNCTIONS TO USE ----- //
// ----------------------------------------------- //



// ----------------------------------------- //
// ----- THE MAIN FUNCTION STARTS HERE ----- //
// ----------------------------------------- //
/**
* This is the main function of the program. Here we will read a .csv file and
* then apply the single neuron in Deep Neural Network on the input data
* contained in it and a generated output data by applying such algorithm with
* CPU parallelism through POSIX Threads. In addition, some evaluation metrics
* will be applied to evaluate the model. Finally, the results will be saved in
* two new .csv files and in a .png file for further comparison and validation
* purposes.
*
* @param int argc - This argument will posses the length number of what is
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
	char csv1Directory[] = "../../../../databases/regression/logisticEquationSystem/1000systems_1000samplesPerSys.csv"; // Directory of the reference .csv file
	char nameOfTheCsvFile1[] = "CenyML_logisticRegression_Coefficients.csv"; // Name the .csv file that will store the resulting coefficient values.
	char nameOfTheCsvFile2[] = "CenyML_logisticRegression_EvalMetrics.csv"; // Name the .csv file that will store the resulting evaluation metrics for the ML model to be obtained.
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file (which is declared in "csvManager.h").
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	int columnIndexOfOutputDataInCsvFile = 2; // This variable will contain the index of the first column in which we will specify the location of the real output values (Y).
	int columnIndexOfInputDataInCsvFile = 3; // This variable will contain the index of the first column in which we will specify the location of the input values (X).
	struct singleNeuronDnnStruct_parallelCPU neuron1; // We create a singleNeuronDnnStruct_parallelCPU structure variable to manage the data input and output data of the single neuron in DNN that will be created.
	neuron1.cpuThreads = 15; // This variable will define the number of CPU threads that wants to be used to parallelize the training and predictions made by the neuron to be created.
	neuron1.m = 1; // This variable will contain the number of features (independent variables) that the input matrix is expected to have.
	neuron1.p = 1; // This variable will contain the number of outputs that the output matrix is expected to have.
	neuron1.isInitial_w = 1; // This variable will indicate whether or not initial values will be given by the implementer (with value of 1) or if random ones are going to be used (with value of 0).
	neuron1.w_first = (double *) malloc((neuron1.m+1)*sizeof(double)); // We allocate the memory required for the variable "neuron1.w_first", which will store the initial coefficient values of the neuron to be created.
	neuron1.w_first[0] = 0; // We define the customized desired value for the bias of the neuron to be created.
	neuron1.w_first[1] = 0; // We define the customized desired value for the weight_1 value of the neuron to be created.
	neuron1.isClassification = 0; // This variable will indicate whether or not it is desired that the neuron considers the input data for a classification (with a vlaue of 1) or a regression problem (with a value of 0).
	//neuron1.threshold = 0.5; // This variable will be used to store the desired threshold value to be used in classification problems by the neuron to be created.
	//neuron1.desiredValueForGroup1 = 1; // This variable will be used to store the label to be used for the group 1 in classification problems by the neuron to be created.
	//neuron1.desiredValueForGroup2 = -1; // This variable will be used to store the label to be used for the group 2 in classification problems by the neuron to be created.
	neuron1.activationFunctionToBeUsed = 2; // This variable tells the neuron what activation function to use (see the commented documentation in the function "getSingleNeuronDNN()" for more details).
	neuron1.learningRate = 0.00000001; // This variable stores the desired learning rate for the neuron to be created.
	neuron1.stopAboveThisAccuracy = 0.95; // The value of this variable is used as a stop function for the single neuron in DNN learning procces.
	neuron1.maxEpochs = 40000; // This variable stores the desired value for the maximum permitted epochs for the training process of the neuron.
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
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to initialize the input data to be used.
	// Allocate the memory required for the variable "neuron1.Y", which will contain the real output data of the system under study.
	neuron1.Y = (double *) malloc(neuron1.n*neuron1.p*sizeof(double));
	// Allocate the memory required for the variable "neuron1.X", which will contain the input data of the system under study.
	neuron1.X = (double *) malloc(neuron1.n*neuron1.m*sizeof(double));
	// Create the output (neuron1.Y) and input (neuron1.X) data with the same rows as in the reference .csv file and their corresponding number of columns.
	for (int currentRow=0; currentRow<neuron1.n; currentRow++) { // Since neuron1.m=neuron1.p=1 for this particular ML algorithm, both "neuron1.Y" and "neuron1.X" will be initialized here.
		neuron1.Y[currentRow] = csv1.allData[columnIndexOfOutputDataInCsvFile + currentRow*databaseColumns1];
		neuron1.X[currentRow] = csv1.allData[columnIndexOfInputDataInCsvFile + currentRow*databaseColumns1];
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to initialize the input data to be used.
	printf("Output and input data initialization elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------------------- DATA MODELING ----------------------- //
	printf("Initializing CenyML single neuron in Deep Neural Network algorithm ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to apply the single neuron in Deep Neural Network with the input data (neuron1.X).
	neuron1.w_best = (double *) malloc((neuron1.m+1)*sizeof(double)); // We allocate the memory required for the variable "neuron1.w_best", which will store the best coefficient values identified by the neuron to be created, after its training process.
	neuron1.w_new = (double *) malloc((neuron1.m+1)*sizeof(double)); // We allocate the memory required for the variable "neuron1.w_new", which will store the last coefficient values identified by the neuron to be created, after its training process.
	// We apply the single neuron in Deep Neural Network algorithm with respect to the input matrix "neuron1.X" and the result is stored in the memory location of the pointer "b".
	getSingleNeuronDNN_parallelCPU(&neuron1);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to apply the single neuron in Deep Neural Network with the input data (neuron1.X).
	printf("CenyML single neuron in Deep Neural Network algorithm elapsed %f seconds.\n\n", elapsedTime);
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	// We predict the input values (neuron1.X) with the machine learning model that was obtained.
	printf("Initializing CenyML predictions with the model that was obtained ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to apply the prediction with the model that was obtained.
	// Allocate the memory required for the variable "Y_hat", which will contain the predicted output data of the system under study.
	double *Y_hat = (double *) malloc(neuron1.n*sizeof(double));
	// We obtain the predicted values with the machine learning model that was obtained.
	predictSingleNeuronDNN_parallelCPU(&neuron1, Y_hat);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the prediction wit hthe model that was obtained.
	printf("The CenyML predictions with the model that was obtained elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the mean squared error metric.
	printf("Initializing CenyML mean squared error metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the mean squared error metric between "neuron1.Y" and "Y_hat".
	// Allocate the memory required for the variable "MSE" (which will contain the results of the mean squared error metric between "neuron1.Y" and "Y_hat").
	double *MSE = (double *) calloc(1, sizeof(double));
	// We apply the mean squared error metric between "neuron1.Y" and "Y_hat".
	getMeanSquaredError(neuron1.Y, Y_hat, neuron1.n, neuron1.m, -neuron1.m, MSE);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the mean squared error metric between "neuron1.Y" and "Y_hat".
	printf("CenyML mean squared error metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the coefficient of determination metric.
	printf("Initializing CenyML coefficient of determination metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the coefficient of determination metric between "neuron1.Y" and "Y_hat".
	// Allocate the memory required for the variable "Rsquared" (which will contain the results of the coefficient of determination metric between "neuron1.Y" and "Y_hat").
	double *Rsquared = (double *) calloc(1, sizeof(double));
	// We apply the coefficient of determination metric between "neuron1.Y" and "Y_hat".
	getCoefficientOfDetermination(neuron1.Y, Y_hat, neuron1.n, Rsquared);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the coefficient of determination metric between "neuron1.Y" and "Y_hat".
	printf("CenyML coefficient of determination metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the adjusted coefficient of determination metric.
	printf("Initializing CenyML adjusted coefficient of determination metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the adjusted coefficient of determination metric between "neuron1.Y" and "Y_hat".
	// Allocate the memory required for the variable "adjustedRsquared" (which will contain the results of the adjusted coefficient of determination metric between "neuron1.Y" and "Y_hat").
	double *adjustedRsquared = (double *) calloc(1, sizeof(double));
	// We apply the adjusted coefficient of determination metric between "neuron1.Y" and "Y_hat".
	getAdjustedCoefficientOfDetermination(neuron1.Y, Y_hat, neuron1.n, neuron1.m, 1, adjustedRsquared);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the adjusted coefficient of determination metric between "neuron1.Y" and "Y_hat".
	printf("CenyML adjusted coefficient of determination metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We create a single variable that contains within all the evaluation metrics that were tested.
	printf("Initializing single variable that will store all the evaluation metrics done ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the adjusted coefficient of determination metric between "neuron1.Y" and "Y_hat".
	// Allocate the memory required for the variable "evaluationMetrics" (which will contain all the results of the evaluation metrics that were obtained).
	double *evaluationMetrics = (double *) malloc(3*sizeof(double));
	evaluationMetrics[0] = MSE[0]; // We add the mean squared error metric.
	evaluationMetrics[1] = Rsquared[0]; // We add the coefficient of determination metric.
	evaluationMetrics[2] = adjustedRsquared[0]; // We add the adjusted coefficient of determination metric.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the adjusted coefficient of determination metric between "neuron1.Y" and "Y_hat".
	printf("Initialization of single variable to store all the evaluation metrics elapsed %f seconds.\n\n", elapsedTime);
	
	// We store the coefficients that were obtained.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .csv file which will store the coefficients that were obtained.
	// Define the desired header names for the new .csv file to be create.
    char csvHeaders1[strlen("coefficients")+1]; // Variable where the following code will store the .csv headers.
    csvHeaders1[0] = '\0'; // Initialize this char variable with a null value.
	strcat(csvHeaders1, "coefficients"); // We add the headers into "csvHeaders".
	// Create a new .csv file and save the results obtained in it.
	char is_nArray1 = 0; // Indicate through this flag variable that the variable that indicates the samples (neuron1.n) is not an array because it has the same amount of samples per columns.
	char isInsertId1 = 0; // Indicate through this flag variable that it is not desired that the file to be created automatically adds an "id" to each row.
	int csvFile_n1 = 2; // This variable is used to indicate the number of rows with data that will be printed in the .csv file to be created.
	createCsvFile(nameOfTheCsvFile1, csvHeaders1, neuron1.w_best, &csvFile_n1, is_nArray1, 1, isInsertId1); // We create the desired .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Creation of the .csv file to store the coefficients that were obtained, elapsed %f seconds.\n\n", elapsedTime);
	
	// We store the resulting evaluation metrics that were obtained.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .csv file which will store the results of the evaluation metrics that were obtained.
	// Define the desired header names for the new .csv file to be create.
    char csvHeaders2[strlen("MSE, Rsquared, adjustedRsquared")+1]; // Variable where the following code will store the .csv headers.
    csvHeaders2[0] = '\0'; // Initialize this char variable with a null value.
	strcat(csvHeaders2, "MSE, Rsquared, adjustedRsquared"); // We add the headers into "csvHeaders".
	// Create a new .csv file and save the results obtained in it.
	char is_nArray2 = 0; // Indicate through this flag variable that the variable that indicates the samples (1) is not an array because it has the same amount of samples per columns.
	char isInsertId2 = 0; // Indicate through this flag variable that it is not desired that the file to be created automatically adds an "id" to each row.
	int csvFile_n2 = 1; // This variable is used to indicate the number of rows with data that will be printed in the .csv file to be created.
	createCsvFile(nameOfTheCsvFile2, csvHeaders2, evaluationMetrics, &csvFile_n2, is_nArray2, 3, isInsertId2); // We create the desired .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Creation of the .csv file to store the evaluation metrics that were obtained, elapsed %f seconds.\n\n", elapsedTime);
	
	// Plot a graph with the model that was obtained and saved it into a .png file.
	printf("Initializing creation of .png image to store the plot of the predicted data and the actual data ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .png file that will store the results of the predicted and actual data.
	// Trying the "pbPlots" library (https://github.com/InductiveComputerScience/pbPlots)
	_Bool success;
    StringReference *errorMessage;
	RGBABitmapImageReference *imageReference = CreateRGBABitmapImageReference();

	ScatterPlotSeries *series = GetDefaultScatterPlotSeriesSettings();
	series->xs = neuron1.X;
	series->xsLength = neuron1.n;
	series->ys = neuron1.Y;
	series->ysLength = neuron1.n;
	series->linearInterpolation = false;
	series->pointType = L"dots";
	series->pointTypeLength = wcslen(series->pointType);
	series->color = CreateRGBColor(0.929, 0.196, 0.216);

	ScatterPlotSeries *series2 = GetDefaultScatterPlotSeriesSettings();
	series2->xs = neuron1.X;
	series2->xsLength = neuron1.n;
	series2->ys = Y_hat;
	series2->ysLength = neuron1.n;
	series2->linearInterpolation = true;
	series2->lineType = L"solid";
	series2->lineTypeLength = wcslen(series->lineType);
	series2->lineThickness = 2;
	series2->color = CreateRGBColor(0.153, 0.153, 0.996);

	ScatterPlotSettings *settings = GetDefaultScatterPlotSettings();
	settings->width = 600;
	settings->height = 400;
	settings->autoBoundaries = true;
	settings->autoPadding = true;
	settings->title = L"";
	settings->titleLength = wcslen(settings->title);
	settings->xLabel = L"independent variable";
	settings->xLabelLength = wcslen(settings->xLabel);
	settings->yLabel = L"dependent variable";
	settings->yLabelLength = wcslen(settings->yLabel);
	ScatterPlotSeries *s [] = {series, series2};
	settings->scatterPlotSeries = s;
	settings->scatterPlotSeriesLength = 2;

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
	printf("Initialization of the creation of the .png file elapsed %f seconds.\n\n", elapsedTime);
	printf("The program has been successfully completed!\n");
	
	
	// Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
	free(neuron1.w_first);
	free(csv1.rowsAndColumnsDimensions);
	free(csv1.allData);
	free(neuron1.Y);
	free(neuron1.X);
	free(neuron1.w_best);
	free(neuron1.w_new);
	free(Y_hat);
	free(MSE);
	free(Rsquared);
	free(adjustedRsquared);
	free(evaluationMetrics);
	free(errorMessage);
	return (0); // end of program.
}

