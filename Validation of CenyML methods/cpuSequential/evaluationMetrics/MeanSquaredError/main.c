/*
* This program will read two different .csv files, which are the ones for
* the linear equation system for both with and without the random bias
* value. Then all their output data will be extracted and saved into the
* matrix "Y" (for the real data) and "Y_hat" (for the predicted data).
* Subsequently, the evaluation metric known as the mean squared error, will
* be applied with respect to both of these output matrixes. Finally, a new
* .csv file "CenyML_getMeanSquaredError_Results.csv" will be created and in
* it, the resulting values of applying the mean squared error metric will
* be saved for further comparations and validations purposes.
*/

 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
#include "../../../../CenyML library skeleton/otherLibraries/time/mTimeTer.h" // library to count the time elapsed.
#include "../../../../CenyML library skeleton/otherLibraries/csv/csvManager.h" // library to open and create .csv files.
#include "../../../../CenyML library skeleton/CenyML_Library/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.h" // library to use the regression evaluation metrics of CenyML.


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
* This is the main function of the program. Here we will read two .csv files and
* then calculate the mean squared error metric between their output values.
* Finally, the results of the applied metric will be saved in a new .csv file for
* further comparation and validation purposes. 
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
* CREATION DATE: NOVEMBER 09, 2021
* LAST UPDATE: NOVEMBER 23, 2021
*/
int main(int argc, char **argv) {
	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	char csv1Directory[] = "../../../../databases/regressionDBs/randLinearEquationSystem/1000systems_1000samplesPerSys.csv"; // Directory of the .csv file containing the real data of the system under study.
	char csv2Directory[] = "../../../../databases/regressionDBs/linearEquationSystem/1000systems_1000samplesPerSys.csv"; // Directory of the .csv file containing the predicted data of the system under study.
	char nameOfTheCsvFile[] = "CenyML_getMeanSquaredError_Results.csv"; // Name the .csv file that will store the results.
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file containing the real data of the system under study (which is declared in "csvManager.h").
	struct csvManager csv2; // We create a csvManager structure variable to manage the desired .csv file containing the predicted data of the system under study (which is declared in "csvManager.h").
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	csv2.fileDirectory = csv2Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv2.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	int m = 1; // This variable will contain the number of features or independent variables that the input matrix should have.
	int p = 1; // This variable will contain the number of outputs that the output matrix is expected to have.
	int columnIndexOfOutputDataInCsvFile = 2; // This variable will contain the index of the first column in which we will specify the location of the output values (Y and/or Y_hat).
	// NOTE: The degrees of freedom that will be assigned will be chosen to remove completely its application and errase the effect of "m" because scikit-learn applies this metric without a degree of freedom.
	int degreesOfFreedom = -m; // Desired degrees of freedom to be applied in the evaluation metric to be calculated. A "0" would represent a degrees of freedom of "n", a "1" would represent a "n-1", ..., a "degrees" would represent a "n-degrees".
	
	
	// ---------------------- IMPORT DATA 1 TO BE USED --------------------- //
	printf("Initializing data extraction from .csv file 1 containing the real output data ...\n");
	double startingTime, elapsedTime; // Declaration of variables used to count time in seconds.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to obtain the data from the reference .csv file.
	// Obtain the rows and columns dimensions of the data of the csv file (excluding headers)
	csv1.rowsAndColumnsDimensions = (int *) malloc(2*sizeof(int)); // We initialize the variable that will store the rows & columns dimensions.
	getCsvRowsAndColumnsDimensions(&csv1); // We input the memory location of the "csv1" into the argument of this function to get the rows & columns dimensions.
	// We save the rows and columns dimensions obtained in some variables that relate to the mathematical symbology according to the documentation of the method to be validated.
	int n = csv1.rowsAndColumnsDimensions[0]; // total number of rows of the output matrixes (Y and/or Y_hat).
	int databaseColumns1 = csv1.rowsAndColumnsDimensions[1]; // total number of columns of the database that was opened.
	// From the structure variable "csv1", we allocate the memory required for the variable (csv1.allData) so that we can store the data of the .csv file in it.
	csv1.allData = (double *) malloc(n*databaseColumns1*sizeof(double));
	// We retrieve the data contained in the reference .csv file.
	getCsvFileData(&csv1); // We input the memory location of the "csv1" into the argument of this function to get all the data contained in the .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the data from the reference .csv file.
	printf("Data extraction from .csv file 1 containing %d rows for each of the %d columns (total samples = %d), elapsed %f seconds.\n\n", n, databaseColumns1, (n*databaseColumns1), elapsedTime);
	
	// ------------------ PREPROCESSING OF THE DATA 1 ------------------ //
	printf("Initializing real output data with %d samples for each of the %d columns (total samples = %d)...\n", n, p, (n*p));
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to innitialize the real output data to be used.
	// Allocate the memory required for the variable "Y", which will contain the real output data of the system under study.
	double *Y = (double *) malloc(n*p*sizeof(double));
	// Pass the real output data into the pointer variable "Y".
	for (int currentRow=0; currentRow<n; currentRow++) {
		for (int currentColumn=0; currentColumn<p; currentColumn++) {
			Y[currentColumn + currentRow*p] = csv1.allData[columnIndexOfOutputDataInCsvFile + currentColumn + currentRow*databaseColumns1];
		}
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to innitialize the input data to be used.
	printf("Real output data innitialization elapsed %f seconds.\n\n", elapsedTime);
	
	// ---------------------- IMPORT DATA 2 TO BE USED --------------------- //
	printf("Initializing data extraction from .csv file 2 containing the predicted output data ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to obtain the data from the reference .csv file.
	// Obtain the rows and columns dimensions of the data of the csv file (excluding headers)
	csv2.rowsAndColumnsDimensions = (int *) malloc(2*sizeof(int)); // We initialize the variable that will store the rows & columns dimensions.
	getCsvRowsAndColumnsDimensions(&csv2); // We input the memory location of the "csv2" into the argument of this function to get the rows & columns dimensions.
	int databaseColumns2 = csv2.rowsAndColumnsDimensions[1]; // total number of columns of the database that was opened.
	// From the structure variable "csv2", we allocate the memory required for the variable (csv2.allData) so that we can store the data of the .csv file in it.
	csv2.allData = (double *) malloc(n*databaseColumns2*sizeof(double));
	// We retrieve the data contained in the reference .csv file.
	getCsvFileData(&csv2); // We input the memory location of the "csv2" into the argument of this function to get all the data contained in the .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the data from the reference .csv file.
	printf("Data extraction from .csv file 2 containing %d rows for each of the %d columns (total samples = %d), elapsed %f seconds.\n\n", n, databaseColumns2, (n*databaseColumns2), elapsedTime);
	
	// ------------------ PREPROCESSING OF THE DATA 2 ------------------ //
	printf("Initializing predicted output data with %d samples for each of the %d columns (total samples = %d)...\n", n, p, (n*p));
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to innitialize the predicted output data to be used.
	// Allocate the memory required for the variable "Y_hat", which will contain the predicted output data of the system under study.
	double *Y_hat = (double *) malloc(n*p*sizeof(double));
	// Pass the predicted output data into the pointer variable "Y_hat".
	for (int currentRow=0; currentRow<n; currentRow++) {
		for (int currentColumn=0; currentColumn<p; currentColumn++) {
			Y_hat[currentColumn + currentRow*p] = csv2.allData[columnIndexOfOutputDataInCsvFile + currentColumn + currentRow*databaseColumns2];
		}
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to innitialize the input data to be used.
	printf("Predicted output data innitialization elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------------------- DATA MODELING ----------------------- //
	// We apply the mean squared error metric.
	printf("Initializing CenyML mean squared error metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the mean squared error metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "MSE" (which will contain the results of the mean squared error metric between "Y" and "Y_hat").
	double *MSE = (double *) calloc(p, sizeof(double));
	// We apply the mean squared error metric between "Y" and "Y_hat".
	getMeanSquaredError(Y, Y_hat, n, m, p, degreesOfFreedom, MSE);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the mean squared error metric between "Y" and "Y_hat".
	printf("CenyML mean squared error metric elapsed %f seconds.\n\n", elapsedTime);
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	// We save the results of the applied mean squared error metric.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .csv file which will store the results that were obtained.
	// Define the desired header names for the new .csv file to be create.
    char csvHeaders[strlen("MSE_result") + 1]; // Variable where the following code will store the .csv headers.
    csvHeaders[0] = '\0'; // Innitialize this char variable with a null value.
	strcat(csvHeaders, "MSE_result"); // We add the column headers into "csvHeaders".
	// Create a new .csv file and save the results obtained in it.
	char is_nArray = 0; // Indicate through this flag variable that the variable that indicates the samples (1) is not an array because it has the same amount of samples per columns.
	char isInsertId = 0; // Indicate through this flag variable that it is not desired that the file to be created automatically adds an "id" to each row.
	int csvFile_n = 1; // This variable is used to indicate the number of rows with data that will be printed in the .csv file to be created.
	createCsvFile(nameOfTheCsvFile, csvHeaders, MSE, &csvFile_n, is_nArray, p, isInsertId); // We create the desired .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Creation of the .csv file to store the results obtained, elapsed %f seconds.\n\n", elapsedTime);
	printf("The program has been successfully completed!");
	
	
	// Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
	free(csv1.rowsAndColumnsDimensions);
	free(csv1.allData);
	free(csv2.rowsAndColumnsDimensions);
	free(csv2.allData);
	free(Y);
	free(Y_hat);
	free(MSE);
	return (0); // end of program.
}

