/*
* This program will read three different .csv files: 1) One that will
* contain the real output values; 2) Another one containing the results
* obtained in Python with the comparative/reference library and; 3) A
* last one containing the predicted output data. Then, all their data
* will be extracted and saved into the following variables: 1) Y; 2) 
* python_adjustedRsquared and; 3) Y_hat. Subsequently, the evaluation
* metric known as the adjusted coefficient of determination will be
* applied with respect to the two output matrixes "Y" and "Y_hat".
* Finally, the result obtained will be compared with the one obtained
* in Python and whether the result ended in a match or not, it will be
* shown in the terminal in which this program was excecuted.
*/

 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../../../CenyML library skeleton/otherLibraries/time/mTime.h" // library to count the time elapsed in Linux Ubuntu.
//#include "../../../../CenyML library skeleton/otherLibraries/time/mTimeTer.h" // library to count the time elapsed in Cygwin terminal window.#include "../../../../CenyML library skeleton/otherLibraries/time/mTimeTer.h" // library to count the time elapsed.
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
* This is the main function of the program. Here we will three .csv files and
* then calculate the coefficient of determination metric between two output
* matrixes. Finally, the results of the applied metric will be compared with
* the ones obtained in Python (this result is stored in one of those three
* .csv files) for validation purposes.
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
* CREATION DATE: NOVEMBER 12, 2021
* LAST UPDATE: DECEMBER 04, 2021
*/
int main(int argc, char **argv) {
	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	char csv1Directory[] = "../../../../databases/regression/randMultipleLinearSystem/100systems_100samplesPerAxisPerSys.csv"; // Directory of the .csv file containing the real data of the system under study.
	char csv2Directory[] = "adjustedRsquared_results.csv"; // Directory of the .csv file containing the results obtained in the comparative library used in Python.
	char csv3Directory[] = "adjustedRsquared_predictedData.csv"; // Directory of the .csv file containing the predicted data of the system under study.
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file containing the real data of the system under study (which is declared in "csvManager.h").
	struct csvManager csv2; // We create a csvManager structure variable to manage the desired .csv file containing the results obtained in Python (which is declared in "csvManager.h").
	struct csvManager csv3; // We create a csvManager structure variable to manage the desired .csv file containing the predicted data of the system under study (which is declared in "csvManager.h").
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	csv2.fileDirectory = csv2Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv2.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	csv3.fileDirectory = csv3Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv3.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	int m = 2; // This variable will contain the number of features (independent variables) that the input matrix is expected to have.
	int p = 1; // This variable will contain the number of outputs that the output matrix is expected to have.
	int columnIndexOfOutputDataInCsvFile = 2; // This variable will contain the index of the first column in which we will specify the location of the real output values (Y).
	int degreesOfFreedom = 1; // Desired degrees of freedom, specified with the variable "q" in the documentation, to be applied in the adjusted R-squared to be calculated.
	
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
	printf("Initializing data extraction from .csv file 2 containing the results to be used as a reference ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to obtain the data from the reference .csv file.
	// Obtain the rows and columns dimensions of the data of the csv file (excluding headers)
	csv2.rowsAndColumnsDimensions = (int *) malloc(2*sizeof(int)); // We initialize the variable that will store the rows & columns dimensions.
	getCsvRowsAndColumnsDimensions(&csv2); // We input the memory location of the "csv2" into the argument of this function to get the rows & columns dimensions.
	// We save the rows and columns dimensions obtained in some variables that relate to the mathematical symbology according to the documentation of the method to be validated.
	int databaseRows2 = csv2.rowsAndColumnsDimensions[0]; // total number of expected rows for the results obtained.
	int databaseColumns2 = csv2.rowsAndColumnsDimensions[1]; // total number of columns of the database that was opened.
	// From the structure variable "csv2", we allocate the memory required for the variable (csv2.allData) so that we can store the data of the .csv file in it.
	csv2.allData = (double *) malloc(databaseRows2*databaseColumns2*sizeof(double));
	// We retrieve the data contained in the reference .csv file.
	getCsvFileData(&csv2); // We input the memory location of the "csv2" into the argument of this function to get all the data contained in the .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the data from the reference .csv file.
	printf("Data extraction from .csv file 2 containing %d rows for each of the %d columns (total samples = %d), elapsed %f seconds.\n\n", databaseRows2, databaseColumns2, (databaseRows2*databaseColumns2), elapsedTime);
	
	// ------------------ PREPROCESSING OF THE DATA 2 ------------------ //
	printf("Initializing Python results data with %d samples for each of the %d columns (total samples = %d)...\n", databaseRows2, databaseColumns2, (databaseRows2*databaseColumns2));
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to innitialize the Python results data to be used.
	// Allocate the memory required for the variable "python_adjustedRsquared", which will contain the results that were obtained in the reference library on Python.
	double *python_adjustedRsquared = (double *) malloc(databaseRows2*databaseColumns2*sizeof(double));
	// Pass the Python results data into the pointer variable "python_adjustedRsquared".
	for (int currentRow=0; currentRow<databaseRows2; currentRow++) {
		for (int currentColumn=0; currentColumn<databaseColumns2; currentColumn++) {
			python_adjustedRsquared[currentColumn + currentRow*databaseColumns2] = csv2.allData[currentColumn + currentRow*databaseColumns2];
		}
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to innitialize the Python results data to be used.
	printf("Python results data innitialization elapsed %f seconds.\n\n", elapsedTime);
	
	// ---------------------- IMPORT DATA 3 TO BE USED --------------------- //
	printf("Initializing data extraction from .csv file 3 containing the predicted output data ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to obtain the data from the reference .csv file.
	// Obtain the rows and columns dimensions of the data of the csv file (excluding headers)
	csv3.rowsAndColumnsDimensions = (int *) malloc(2*sizeof(int)); // We initialize the variable that will store the rows & columns dimensions.
	getCsvRowsAndColumnsDimensions(&csv3); // We input the memory location of the "csv3" into the argument of this function to get the rows & columns dimensions.
	int databaseColumns3 = csv3.rowsAndColumnsDimensions[1]; // total number of columns of the database that was opened.
	// From the structure variable "csv3", we allocate the memory required for the variable (csv3.allData) so that we can store the data of the .csv file in it.
	csv3.allData = (double *) malloc(n*databaseColumns3*sizeof(double));
	// We retrieve the data contained in the reference .csv file.
	getCsvFileData(&csv3); // We input the memory location of the "csv3" into the argument of this function to get all the data contained in the .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the data from the reference .csv file.
	printf("Data extraction from .csv file 3 containing %d rows for each of the %d columns (total samples = %d), elapsed %f seconds.\n\n", n, databaseColumns2, (n*databaseColumns2), elapsedTime);
	
	// ------------------ PREPROCESSING OF THE DATA 3 ------------------ //
	printf("Initializing predicted output data with %d samples for each of the %d columns (total samples = %d)...\n", n, p, (n*p));
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to innitialize the predicted output data to be used.
	// Allocate the memory required for the variable "Y_hat", which will contain the predicted output data of the system under study.
	double *Y_hat = (double *) malloc(n*p*sizeof(double));
	// Pass the predicted output data into the pointer variable "Y_hat".
	for (int currentRow=0; currentRow<n; currentRow++) {
		for (int currentColumn=0; currentColumn<p; currentColumn++) {
			Y_hat[currentColumn + currentRow*p] = csv3.allData[currentColumn + currentRow*databaseColumns3];
		}
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to innitialize the input data to be used.
	printf("Predicted output data innitialization elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------------------- DATA MODELING ----------------------- //
	// We apply the adjusted coefficient of determination metric.
	printf("Initializing CenyML adjusted coefficient of determination metric ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the adjusted coefficient of determination metric between "Y" and "Y_hat".
	// Allocate the memory required for the variable "adjustedRsquared" (which will contain the results of the adjusted coefficient of determination metric between "Y" and "Y_hat").
	double *adjustedRsquared = (double *) calloc(p, sizeof(double));
	// We apply the adjusted coefficient of determination metric between "Y" and "Y_hat".
	getAdjustedCoefficientOfDetermination(Y, Y_hat, n, m, degreesOfFreedom, adjustedRsquared);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the adjusted coefficient of determination metric between "Y" and "Y_hat".
	printf("CenyML adjusted coefficient of determination metric elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	// We validate the CenyML adjusted coefficient of determination metric.
	printf("Initializing CenyML validation process with respect to the results obtained in Python ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .csv file which will store the results that were obtained.
	double differentiation; // Variable used to store the error obtained for a certain value.
	double epsilon = 1.0E-14; // Variable used to store the max error value permitted during validation process.
	char isMatch = 1; // Variable used as a flag to indicate if the current comparation of values stands for a match. Note that the value of 1 = is a match and 0 = is not a match.
	// We check that all the differentiations do not surpass the error indicated through the variable "epsilon".
	for (int currentRow=0; currentRow<databaseRows2; currentRow++) {
		for (int currentColumn=0; currentColumn<p; currentColumn++) {
			differentiation = fabs(adjustedRsquared[currentColumn + currentRow*p] - python_adjustedRsquared[currentColumn + currentRow*p]);
			if (differentiation > epsilon) { // if the error surpassed the value permitted, then terminate validation process and emit message to indicate a non match.
				isMatch = 0;
				printf("Validation process DID NOT MATCH! and a difference of %f was obtained.\n", differentiation);
				break;
			}
		}
		if (isMatch == 0) {
			break;
		}
	}
	if (isMatch == 1) { // If the flag "isMatch" indicates a true/high value, then emit message to indicate that the validation process matched.
		printf("Validation process MATCHED!\n");
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Validation of the results obtained, elapsed %f seconds.\n\n", elapsedTime);
	printf("The program has been successfully completed!\n");
	
	
	// Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
	free(csv1.rowsAndColumnsDimensions);
	free(csv1.allData);
	free(csv2.rowsAndColumnsDimensions);
	free(csv2.allData);
	free(csv3.rowsAndColumnsDimensions);
	free(csv3.allData);
	free(Y);
	free(python_adjustedRsquared);
	free(Y_hat);
	free(adjustedRsquared);
	return (0); // end of program.
}

