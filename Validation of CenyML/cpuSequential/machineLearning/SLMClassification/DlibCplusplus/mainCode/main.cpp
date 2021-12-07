/*
* This program will read a .csv file containing the data of a linear
* classification equation system to then exctact all its data. Its input
* data will be saved into the matrix "X" and its output data into the
* matrix "Y". Subsequently, a linear support vector machine
* classification method will be applied to obtain the best fitting
* model for such data. Then, some evaluation metrics will be applied.
* Next, a new .csv file will be created to save the results obtained
* with the evaluation metrics. Finally, a plot of the predicted data by
* the obtained model with respect to the actual data, will be plotted
* and saved into a .png file. Both the .csv file and this .png file
* will serve for further comparations and validation purposes.
*
* URL TO DOWNLOAD THE DLIB LIBRARY VERSION 19.22:
* https://bit.ly/3DS3fBx
*
* DOCUMENTATION TO LEARN HOW TO COMPILE THE DLIB LIBRARY:
* https://bit.ly/3xmF2k5
* NOTE: I compiled on Linux From Command Line (see the makefile that is located under the same directory as this file).
*
* DOCUMENTATION TO LEARN ABOUT THE C FORMULATION OF A SUPPORT VECTOR MACHINE AS PROVIDED BY THE DLIB LIBRARY:
* https://bit.ly/3xwDrYY
*/

// ------------------------------------------------- //
// ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
// ------------------------------------------------- //
#include <iostream>
#include <ctime>
#include <vector>
#include <dlib/svm.h> // Dlib library version 19.22
#include <stdio.h>
#include <stdlib.h>
#include "../../../../../../CenyML library skeleton/otherLibraries/time/mTime.h" // library to count the time elapsed in Linux Ubuntu.
//#include "../../../../../../CenyML library skeleton/otherLibraries/time/mTimeTer.h" // library to count the time elapsed in Cygwin terminal window.
#include "../../../../../../CenyML library skeleton/otherLibraries/csv/csvManager.h" // library to open and create .csv files.
#include "../../../../../../CenyML library skeleton/otherLibraries/pbPlots/pbPlots.h" // library to generate plots v0.1.9.0
#include "../../../../../../CenyML library skeleton/otherLibraries/pbPlots/supportLib.h"  // library required for "pbPlots.h" v0.1.9.0
#include "../../../../../../CenyML library skeleton/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.h" // library to use the classification evaluation metrics of CenyML.


// ---------------------------------------------- //
// ----- DEFINE THE GLOBAL RESOURCES TO USE ----- //
// ---------------------------------------------- //
using namespace std;
using namespace dlib;


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
* CREATION DATE: NOVEMBER 23, 2021
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
* then apply the linear support vector machine classification on the input
* and output data contained in it. In addition, some evaluation metrics will
* be applied to evaluate the model. Finally, the results will be saved in a
* new .csv file and in a .png file for further comparation and validation
* purposes.
*
* @return 0
*
* @author Miranda Meza Cesar
* CREATION DATE: NOVEMBER 25, 2021
* LAST UPDATE: DECEMBER 06, 2021
*/
int main() {
    // --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	char csv1Directory[] = "database.csv"; // Directory of the reference .csv file
	char nameOfTheCsvFile2[] = "Dlib_linearSVM_evalMetrics.csv"; // Name the .csv file that will store the resulting evaluation metrics for the ML model to be obtained.
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file (which is declared in "csvManager.h").
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	int m = 2; // This variable will contain the number of features (independent variables) that the input matrix is expected to have.
	int p = 1; // This variable will contain the number of outputs that the output matrix is expected to have.
	int columnIndexOfOutputDataInCsvFile = 2; // This variable will contain the index of the first column in which we will specify the location of the real output values (Y).
	int columnIndexOfInputDataInCsvFile = 3; // This variable will contain the index of the first column in which we will specify the location of the input values (X).
    typedef std::map<unsigned long,double> sample_type; // DLIB COMMENT: In this example program we will be dealing with feature vectors that are sparse.
    typedef sparse_linear_kernel<sample_type> kernel_type; // DLIB COMMENT: This is a typedef for the type of kernel we are going to use in this example. 
    svm_c_linear_trainer<kernel_type> linear_trainer; // DLIB COMMENT: Lets use the svm trainer specially optimized for the linear_kernel and sparse_linear_kernel.
    // DLIB COMMENT: This trainer solves the "C" formulation of the SVM.  See the documentation for details.
    // NOTE: If you do not define a specific value for "C" with "set_c()", then it
	//       seems to me like Dilb assigns a default value for it, which may not
	//       always produce satisfactory results. Unlike its Kernel SVM function,
	//       which operates iteratively to find the best possible solution when no
	//       "C" value is assigned, here this does not happen and therefore, I
	//       recommend to assign several values to "C" until you get the best
	//       possible results.
    linear_trainer.set_c(1000); // Define the "C" desired value inside the argument of "linear_trainer.set_c()".
    std::vector<sample_type> samples; // This will be used to store the samples to be managed by the machine learning models of the Dlib library.
    std::vector<double> labels; //  This will be used to store the outputs for each of the samples managed by the machine learning models of the Dlib library.
    sample_type sample; // DLIB COMMENT: make an instance of a sample vector so we can use it below.
	
	
	// ---------------------- IMPORT DATA TO USE --------------------- //
	printf("Initializing data extraction from .csv file containing the data to be used ...\n");
	double startingTime, elapsedTime; // Declaration of variables used to count time in seconds.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to obtain the data from the reference .csv file.
	// Obtain the rows and columns dimensions of the data of the csv file (excluding headers)
	csv1.rowsAndColumnsDimensions = (int *) malloc(2*sizeof(int)); // We initialize the variable that will store the rows & columns dimensions.
	getCsvRowsAndColumnsDimensions(&csv1); // We input the memory location of the "csv1" into the argument of this function to get the rows & columns dimensions.
	// We save the rows and columns dimensions obtained in some variables that relate to the mathematical symbology according to the documentation of the method to be validated.
	int n = csv1.rowsAndColumnsDimensions[0]; // total number of rows of the input matrix (X)
	int databaseColumns1 = csv1.rowsAndColumnsDimensions[1]; // total number of columns of the database that was opened.
	// From the structure variable "csv1", we allocate the memory required for the variable (csv1.allData) so that we can store the data of the .csv file in it.
	csv1.allData = (double *) malloc(n*databaseColumns1*sizeof(double));
	// We retrieve the data contained in the reference .csv file.
	getCsvFileData(&csv1); // We input the memory location of the "csv1" into the argument of this function to get all the data contained in the .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the data from the reference .csv file.
	printf("Data extraction from .csv file containing %d samples for each of the %d columns (total samples = %d), elapsed %f seconds.\n\n", n, databaseColumns1, (n*databaseColumns1), elapsedTime);
	
	
	
	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	printf("Initializing the output and input data with %d samples for each of the %d columns (total samples = %d) each...\n", n, m, n);
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to innitialize the input data to be used.
	// Allocate the memory required for the variable "Y" and "Y_tilde", which will contain the real output data of the system under study.
	double *Y = (double *) malloc(n*p*sizeof(double));
	double *Y_tilde = (double *) malloc(n*p*sizeof(double));
	// Store the data that must be contained in the output matrix "Y".
	for (int currentRow=0; currentRow<n; currentRow++) {
		Y[currentRow] = csv1.allData[columnIndexOfOutputDataInCsvFile + currentRow*databaseColumns1];
		if (Y[currentRow] == 0) {
			Y_tilde[currentRow] = -1;
		} else {
			Y_tilde[currentRow] = 1;
		}
	}
	// Allocate the memory required for the variable "X", which will contain the input data of the system under study.
	double *X = (double *) malloc(n*m*sizeof(double));
	// Store the data that must be contained in the input matrix "X".
	for (int currentRow=0; currentRow<n; currentRow++) {
		for (int currentColumn=0; currentColumn<m; currentColumn++) {
			X[currentColumn + currentRow*m] = csv1.allData[columnIndexOfInputDataInCsvFile + currentColumn + currentRow*databaseColumns1];
		}
	}
    // DLIB COMMENT: Pass the input data of the system under study to the Dlib svm trainer.
    //NOTE: In the Dlib library, it seems to me that all samples must be passed row per row, unlike most other libraries in which you can pass the entire matrix of rows and columns at once.
	for (int currentRow=0; currentRow<n; currentRow++) {
		sample.clear(); // DLIB COMMENT: We clear the current data stored in the instance "sample".
		for (int currentColumn=0; currentColumn<m; currentColumn++) {
			sample[currentColumn] = X[currentColumn + currentRow*m]; // We pass the current row of data from the input matrix "X" to the instance "sample".
		}
		samples.push_back(sample); // DLIB COMMENT: Save the current sample of the input matrix "X" so we can let the svm_c_linear_trainer learn from them below.
		labels.push_back(Y_tilde[currentRow]); // DLIB COMMENT: Save the output of the current sample of the input matrix "X" so we can let the svm_c_linear_trainer learn from them below.
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to innitialize the input data to be used.
	printf("Output and input data innitialization elapsed %f seconds.\n\n", elapsedTime);
	
	
	
	// ------------------------- DATA MODELING ----------------------- //
	printf("Initializing Dlib linear support vector machine classification algorithm ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to apply the simple linear machine classification with the input data (X).	
	decision_function<kernel_type> linearSVM_model = linear_trainer.train(samples, labels); // We train the linear SVM model.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to apply the simple linear machine classification with the input data (X).
	printf("Dlib linear support vector machine classification algorithm elapsed %f seconds.\n\n", elapsedTime);
	
	
	
	// ----------- PREDICTIONS AND EVALUATIONS OF THE MODEL --------- //
	// We predict the input values (X) with the machine learning model that was obtained.
	printf("Initializing Dlib predictions with the model that was obtained ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to apply the prediction with the model that was obtained.
	// Allocate the memory required for the variable "Y_hat", which will contain the predicted output data of the system under study.
	double *Y_hat = (double *) malloc(n*p*sizeof(double));
	// Pass the input data of the system under study to the trained model to make their corresponding predictions and store them.
    //NOTE: In the Dlib library, all samples must be passed row per row, unlike most other libraries in which you can pass the entire matrix of rows and columns at once.
	for (int currentRow=0; currentRow<n; currentRow++) {
		sample.clear(); // We clear the current data stored in the instance "sample".
		for (int currentColumn=0; currentColumn<m; currentColumn++) {
			sample[currentColumn] = X[currentColumn + currentRow*m]; // We pass the current row of data from the input matrix "X" to the instance "sample".
		}
		if (linearSVM_model(sample) > 0) {
			Y_hat[currentRow] = 1;
		} else {
			Y_hat[currentRow] = 0; // Instead of storing a "-1" value, we will store a "0" instead so that we can use the predicted values in the CenyML evaluation metric funtions.
		}
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the prediction wit hthe model that was obtained.
	printf("The Dlib predictions with the model that was obtained elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the cross entropy error metric.
	double NLLepsilon = 1.0E-15; // This variable will contain the user desired epsilon value to be summed to any zero value and substracted to any value of the output matrixes (Y and/or Y_hat). NOTE: It will be assigned the value to match the one used in scikit-learn.
	printf("Initializing CenyML cross entropy error metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the cross entropy error metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "NLL" (which will contain the results of the cross entropy error metric between "Y" and "Y_hat").
	double *NLL = (double *) calloc(1, sizeof(double));
	// We apply the cross entropy error metric between "Y" and "Y_hat".
	getCrossEntropyError(Y, Y_hat, n, NLLepsilon, NLL);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the cross entropy error metric between "Y" and "Y_hat".
	printf("CenyML cross entropy error metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the confusion matrix metric.
	printf("Initializing CenyML confusion matrix metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the confusion matrix metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "confusionMatrix" (which will contain the results of the confusion matrix metric between "Y" and "Y_hat").
	double *confusionMatrix = (double *) calloc(4, sizeof(double));
	// We apply the confusion matrix metric between "Y" and "Y_hat".
	getConfusionMatrix(Y, Y_hat, n, confusionMatrix);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the confusion matrix metric between "Y" and "Y_hat".
	printf("CenyML confusion matrix metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the accuracy metric.
	printf("Initializing CenyML accuracy metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the accuracy metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "accuracy" (which will contain the results of the accuracy metric between "Y" and "Y_hat").
	double *accuracy = (double *) calloc(1, sizeof(double));
	// We apply the accuracy metric between "Y" and "Y_hat".
	getAccuracy(Y, Y_hat, n, accuracy);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the accuracy metric between "Y" and "Y_hat".
	printf("CenyML accuracy metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the precision metric.
	printf("Initializing CenyML precision metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the precision metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "precision" (which will contain the results of the precision metric between "Y" and "Y_hat").
	double *precision = (double *) calloc(1, sizeof(double));
	// We apply the precision metric between "Y" and "Y_hat".
	getPrecision(Y, Y_hat, n, precision);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the precision metric between "Y" and "Y_hat".
	printf("CenyML precision metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the recall metric.
	printf("Initializing CenyML recall metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the recall metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "recall" (which will contain the results of the recall metric between "Y" and "Y_hat").
	double *recall = (double *) calloc(1, sizeof(double));
	// We apply the recall metric between "Y" and "Y_hat".
	getRecall(Y, Y_hat, n, recall);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the recall metric between "Y" and "Y_hat".
	printf("CenyML recall metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the F1 score metric.
	printf("Initializing CenyML F1 score metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the F1 score metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "F1 score" (which will contain the results of the F1 score metric between "Y" and "Y_hat").
	double *F1score = (double *) calloc(1, sizeof(double));
	// We apply the F1 score metric between "Y" and "Y_hat".
	getF1score(Y, Y_hat, n, F1score);
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
	
	
	
	// ----------------- VISUALIZATION OF THE MODEL ------------------ //
	// Plot a graph with the model that was obtained and saved it into a .png file.
	printf("Initializing creation of .png image to store the plot of the predicted data and the actual data ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .png file that will store the results of the predicted and actual data.
	// Trying the "pbPlots" library (https://github.com/InductiveComputerScience/pbPlots)
	_Bool success;
    StringReference *errorMessage;
	RGBABitmapImageReference *imageReference = CreateRGBABitmapImageReference();
	
	// In order to continue with the plotting process, identify the minimum and maximum values contained in each machine learning feature.
	double minX1 = X[0];
	double maxX1 = X[0];
	double minX2 = X[1];
	double maxX2 = X[1];
	for (int currentRow=1; currentRow<n; currentRow++) {
		if (X[currentRow*m] < minX1) {
			minX1 = X[currentRow*m];
		}
		if (X[currentRow*m] > maxX1) {
			maxX1 = X[currentRow*m];
		}
		if (X[1 + currentRow*m] < minX2) {
			minX2 = X[1 + currentRow*m];
		}
		if (X[1 + currentRow*m] > maxX2) {
			maxX2 = X[1 + currentRow*m];
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
	double *bg_X = (double *) malloc((n_ofLinearlySpacedArray*n_ofLinearlySpacedArray)*m*sizeof(double)); // Allocate the memory required for the variable "bg_X".
	int currentRow_bg_X=0;
	for (int currentRow1=0; currentRow1<n_ofLinearlySpacedArray; currentRow1++) { // Store the data that must be contained in the input matrix "bg_X".
		for (int currentRow2=0; currentRow2<n_ofLinearlySpacedArray; currentRow2++) {
			bg_X[0 + currentRow_bg_X*m] = X1[currentRow1];
			bg_X[1 + currentRow_bg_X*m] = X2[currentRow2];
			currentRow_bg_X++;
		}
	}
	// In order to continue the plotting process, we obtain the output data that will be used to create the background of the plot to be created.
	double *bg_Y_hat = (double *) malloc((n_ofLinearlySpacedArray*n_ofLinearlySpacedArray)*p*sizeof(double));
	// Pass the input data of the system under study to the trained model to make their corresponding predictions and store them.
    //NOTE: In the Dlib library, all samples must be passed row per row, unlike most other libraries in which you can pass the entire matrix of rows and columns at once.
	for (int currentRow=0; currentRow<(n_ofLinearlySpacedArray*n_ofLinearlySpacedArray); currentRow++) {
		sample.clear(); // We clear the current data stored in the instance "sample".
		for (int currentColumn=0; currentColumn<m; currentColumn++) {
			sample[currentColumn] = bg_X[currentColumn + currentRow*m]; // We pass the current row of data from the input matrix "X" to the instance "sample".
		}
		if (linearSVM_model(sample) > 0) {
			bg_Y_hat[currentRow] = 1;
		} else {
			bg_Y_hat[currentRow] = -1;
		}
	}
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
			bg_X1_1s[currentRow_bg_X_1s] = bg_X[currentRow*m];
			bg_X2_1s[currentRow_bg_X_1s] = bg_X[1 + currentRow*m];
			currentRow_bg_X_1s++;
		} else {
			bg_X1_0s[currentRow_bg_X_0s] = bg_X[currentRow*m];
			bg_X2_0s[currentRow_bg_X_0s] = bg_X[1 + currentRow*m];
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
	for (int currentRow=0; currentRow<n; currentRow++) {
		if (Y[currentRow] == 1) {
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
	for (int currentRow=0; currentRow<n; currentRow++) {
		if (Y[currentRow] == 1) {
			real_X1_1s[currentRow_predicted_X_1s] = X[currentRow*m];
			real_X2_1s[currentRow_predicted_X_1s] = X[1 + currentRow*m];
			currentRow_predicted_X_1s++;
		} else {
			real_X1_0s[currentRow_predicted_X_0s] = X[currentRow*m];
			real_X2_0s[currentRow_predicted_X_0s] = X[1 + currentRow*m];
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
        WriteToFile(pngdata, length, "plotOfMachineLearningModel (Dlib).png");
        DeleteImage(imageReference->image);
	}else{
	    fprintf(stderr, "Error: ");
        for(int i = 0; i < errorMessage->stringLength; i++){
            fprintf(stderr, "%c", errorMessage->string[i]);
        }
        fprintf(stderr, "\n");
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .png file that will store the results of the predicted and actual data.
	printf("Innitialization of the creation of the .png file elapsed %f seconds.\n", elapsedTime);
	printf("NOTE: In regards to the .png image, the horizontal axis stands for the independent variable 1 and the vertical axis for the independent variable 2.\n");
	printf("In addition, the color green stands for an output value of 1 and the color red for an output value of -1.\n");
	printf("Finally, while the background color represents the predicted 1s and -1s that were made by the machine learning model that was just trained, the dots/triangles stand for the true behaviour of the system under study.\n\n");
	printf("The program has been successfully completed!\n");
	
	// Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
	free(csv1.rowsAndColumnsDimensions);
	free(csv1.allData);
	free(Y);
	free(X);
	free(Y_hat);
	free(NLL);
	free(confusionMatrix);
	free(accuracy);
	free(precision);
	free(recall);
	free(F1score);
	free(evaluationMetrics);
	free(X1);
	free(X2);
	free(bg_X);
	free(bg_Y_hat);
	free(bg_X1_1s);
	free(bg_X2_1s);
	free(bg_X1_0s);
	free(bg_X2_0s);
	free(real_X1_1s);
	free(real_X2_1s);
	free(real_X1_0s);
	free(real_X2_0s);
	return (0); // end of program.
}

