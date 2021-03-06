/*
* This program will read a .csv file containing the data of a multiple
* polynomial equation system to then exctact all its data. Its input
* data will be saved into the matrix "X" and its output data into the
* matrix "Y". Subsequently, a multiple polynomial regression method
* will be applied to obtain the best fitting coefficient values of
* such data. Then, some evaluation metrics will be applied. Next, two
* new .csv files will be created to save: 1) the coefficient values
* that were obtained and 2) the results obtained with the evaluation
* metrics. Both of these .csv files will serve for further comparations
* and validation purposes.
 */

 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
#include "../../../../CenyML_library_skeleton/otherLibraries/time/mTime.h" // library to count the time elapsed in Linux Ubuntu.
//#include "../../../../CenyML_library_skeleton/otherLibraries/time/mTimeTer.h" // library to count the time elapsed in Cygwin terminal window.
#include "../../../../CenyML_library_skeleton/otherLibraries/csv/csvManager.h" // library to open and create .csv files.
#include "../../../../CenyML_library_skeleton/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.h" // library to use the regression evaluation metrics of CenyML.
#include "../../../../CenyML_library_skeleton/CenyML_Library/cpuSequential/machineLearning/CenyMLregression.h" // library to use the regression algorithms of CenyML.


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
* then apply the multiple polynomial regression on the input and output data
* contained in it. In addition, some evaluation metrics will be applied to
* evaluate the model. Finally, the results will be saved in two new .csv files
* for further comparation and validation purposes.
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
* CREATION DATE: NOVEMBER 18, 2021
* LAST UPDATE: DECEMBER 06, 2021
*/
int main(int argc, char **argv) {
	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	char csv1Directory[] = "../../../../databases/regression/multiplePolynomialEquationSystem/100systems_100samplesPerAxisPerSys.csv"; // Directory of the reference .csv file
	char nameOfTheCsvFile1[] = "CenyML_getMultiplePolynomialRegression_Coefficients.csv"; // Name the .csv file that will store the resulting coefficient values.
	char nameOfTheCsvFile2[] = "CenyML_getMultiplePolynomialRegression_evalMetrics.csv"; // Name the .csv file that will store the resulting evaluation metrics for the ML model to be obtained.
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file (which is declared in "csvManager.h").
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	int m = 2; // This variable will contain the number of features (independent variables) that the input matrix is expected to have.
	int p = 1; // This variable will contain the number of outputs that the output matrix is expected to have.
	int N = 2; // This variable will contain the value that is desired for the order of degree of the machine learning model to be trained.
	int columnIndexOfOutputDataInCsvFile = 2; // This variable will contain the index of the first column in which we will specify the location of the real output values (Y).
	int columnIndexOfInputDataInCsvFile = 3; // This variable will contain the index of the first column in which we will specify the location of the input values (X).
	double b_ideal[5]; // This variable will be used to contain the ideal coefficient values that the model to be trained should give.
	b_ideal[0] = 82;
	b_ideal[1] = -1.6;
	b_ideal[2] = 0.016;
	b_ideal[3] = -1.28;
	b_ideal[4] = 0.0128;
	
	
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
	// Allocate the memory required for the variable "Y", which will contain the real output data of the system under study.
	double *Y = (double *) malloc(n*p*sizeof(double));
	// Store the data that must be contained in the output matrix "Y".
	for (int currentRow=0; currentRow<n; currentRow++) {
		Y[currentRow] = csv1.allData[columnIndexOfOutputDataInCsvFile + currentRow*databaseColumns1];
	}
	// Allocate the memory required for the variable "X", which will contain the input data of the system under study.
	double *X = (double *) malloc(n*m*sizeof(double));
	// Store the data that must be contained in the input matrix "X".
	for (int currentRow=0; currentRow<n; currentRow++) {
		for (int currentColumn=0; currentColumn<m; currentColumn++) {
			X[currentColumn + currentRow*m] = csv1.allData[columnIndexOfInputDataInCsvFile + currentColumn + currentRow*databaseColumns1];
		}
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to innitialize the input data to be used.
	printf("Output and input data innitialization elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------------------- DATA MODELING ----------------------- //
	printf("Initializing CenyML multiple polynomial regression algorithm ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to apply the multiple polynomial regression with the input data (X).
	// Allocate the memory required for the variable "b", which will contain the identified best fitting coefficient values that will result from the multiple polynomial regression algorithm.
	double *b = (double *) calloc((m*N+1)*p, sizeof(double));
	// We apply the multiple polynomial regression algorithm with respect to the input matrix "X" and the result is stored in the memory location of the pointer "b".
	getMultiplePolynomialRegression(X, Y, n, m, p, N, (char) 0, (char) 0, b);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to apply the multiple polynomial regression with the input data (X).
	printf("CenyML multiple polynomial regression algorithm elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	// We predict the input values (X) with the machine learning model that was obtained.
	printf("Initializing CenyML predictions with the model that was obtained ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to apply the prediction with the model that was obtained.
	// Allocate the memory required for the variable "Y_hat", which will contain the predicted output data of the system under study.
	double *Y_hat = (double *) malloc(n*p*sizeof(double));
	// We obtain the predicted values with the machine learning model that was obtained.
	predictMultiplePolynomialRegression(X, N, (char) 0, b, n, m, p, Y_hat);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the prediction wit hthe model that was obtained.
	printf("The CenyML predictions with the model that was obtained elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the mean squared error metric.
	printf("Initializing CenyML mean squared error metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the mean squared error metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "MSE" (which will contain the results of the mean squared error metric between "Y" and "Y_hat").
	double *MSE = (double *) calloc(p, sizeof(double));
	// We apply the mean squared error metric between "Y" and "Y_hat".
	getMeanSquaredError(Y, Y_hat, n, m, -m, MSE);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the mean squared error metric between "Y" and "Y_hat".
	printf("CenyML mean squared error metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the coefficient of determination metric.
	printf("Initializing CenyML coefficient of determination metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the coefficient of determination metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "Rsquared" (which will contain the results of the coefficient of determination metric between "Y" and "Y_hat").
	double *Rsquared = (double *) calloc(p, sizeof(double));
	// We apply the coefficient of determination metric between "Y" and "Y_hat".
	getCoefficientOfDetermination(Y, Y_hat, n, Rsquared);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the coefficient of determination metric between "Y" and "Y_hat".
	printf("CenyML coefficient of determination metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We apply the adjusted coefficient of determination metric.
	printf("Initializing CenyML adjusted coefficient of determination metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the adjusted coefficient of determination metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "adjustedRsquared" (which will contain the results of the adjusted coefficient of determination metric between "Y" and "Y_hat").
	double *adjustedRsquared = (double *) calloc(p, sizeof(double));
	// We apply the adjusted coefficient of determination metric between "Y" and "Y_hat".
	getAdjustedCoefficientOfDetermination(Y, Y_hat, n, m, 1, adjustedRsquared);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the adjusted coefficient of determination metric between "Y" and "Y_hat".
	printf("CenyML adjusted coefficient of determination metric elapsed %f seconds.\n\n", elapsedTime);
	
	// We create a single variable that contains within all the evaluation metrics that were tested.
	printf("Initializing single variable that will store all the evaluation metrics done ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the adjusted coefficient of determination metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "evaluationMetrics" (which will contain all the results of the evaluation metrics that were obtained).
	double *evaluationMetrics = (double *) malloc(3*p*sizeof(double));
	evaluationMetrics[0] = MSE[0]; // We add the mean squared error metric.
	evaluationMetrics[1] = Rsquared[0]; // We add the coefficient of determination metric.
	evaluationMetrics[2] = adjustedRsquared[0]; // We add the adjusted coefficient of determination metric.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the adjusted coefficient of determination metric between "Y" and "Y_hat".
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
	int csvFile_n1 = m*N+1; // This variable is used to indicate the number of rows with data that will be printed in the .csv file to be created.
	createCsvFile(nameOfTheCsvFile1, csvHeaders1, b, &csvFile_n1, is_nArray1, p, isInsertId1); // We create the desired .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Creation of the .csv file to store the coefficients that were obtained, elapsed %f seconds.\n\n", elapsedTime);
	
	// We store the resulting evaluation metrics that were obtained.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .csv file which will store the results of the evaluation metrics that were obtained.
	// Define the desired header names for the new .csv file to be create.
    char csvHeaders2[strlen("MSE, Rsquared, adjustedRsquared")+1]; // Variable where the following code will store the .csv headers.
    csvHeaders2[0] = '\0'; // Innitialize this char variable with a null value.
	strcat(csvHeaders2, "MSE, Rsquared, adjustedRsquared"); // We add the headers into "csvHeaders".
	// Create a new .csv file and save the results obtained in it.
	char is_nArray2 = 0; // Indicate through this flag variable that the variable that indicates the samples (1) is not an array because it has the same amount of samples per columns.
	char isInsertId2 = 0; // Indicate through this flag variable that it is not desired that the file to be created automatically adds an "id" to each row.
	int csvFile_n2 = 1; // This variable is used to indicate the number of rows with data that will be printed in the .csv file to be created.
	createCsvFile(nameOfTheCsvFile2, csvHeaders2, evaluationMetrics, &csvFile_n2, is_nArray2, 3*p, isInsertId2); // We create the desired .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Creation of the .csv file to store the evaluation metrics that were obtained, elapsed %f seconds.\n\n", elapsedTime);
	
	// We validate the getMultiplePolynomialRegression method.
	printf("Initializing coefficients validation of the CenyML getMultiplePolynomialRegression method ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to validate the getMultiplePolynomialRegression method.
	double differentiation; // Variable used to store the error obtained for a certain value.
	double epsilon = 1.0E-8; // Variable used to store the max error value permitted during validation process.
	char isMatch = 1; // Variable used as a flag to indicate if the current comparation of values stands for a match. Note that the value of 1 = is a match and 0 = is not a match.
	// We check that all the differentiations do not surpass the error indicated through the variable "epsilon".
	for (int currentRow=0; currentRow<(m*N+1); currentRow++) {
		differentiation = fabs(b[currentRow] - b_ideal[currentRow]);
		if (differentiation > epsilon) { // if the error surpassed the value permitted, then terminate validation process and emit message to indicate a non match.
			isMatch = 0;
			printf("Validation process DID NOT MATCH! and a difference of %f was obtained.\n", differentiation);
			break;
		}
	}
	if (isMatch == 1) { // If the flag "isMatch" indicates a true/high value, then emit message to indicate that the validation process matched.
		printf("Validation process MATCHED!\n");
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to validate the getMultiplePolynomialRegression method.
	printf("The coefficients validation of the CenyML getMultiplePolynomialRegression method elapsed %f seconds.\n\n", elapsedTime);
	printf("The program has been successfully completed!\n");
	
	// Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
	free(csv1.rowsAndColumnsDimensions);
	free(csv1.allData);
	free(Y);
	free(X);
	free(b);
	free(Y_hat);
	free(MSE);
	free(Rsquared);
	free(adjustedRsquared);
	free(evaluationMetrics);
	return (0); // end of program.
}

