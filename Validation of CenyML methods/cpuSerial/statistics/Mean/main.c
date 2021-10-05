/*
 * This program will read the .csv file
* "multiplePolynomialEquationSystem_100systems_100samplesPerAxisPerSys.csv"
* to then exctact all its data and save it into the matrix "X". Subsequently,
* the mean of each row will be calculated and stored in "B_x_bar". Finally,
* a new .csv file "CenyML_getMean_Results.csv" will be created and in it, the
* means for each column will be saved for further comparations and validations
* of the mean method.
 */

 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
#include "../../../../CenyML Library/otherLibraries/time/mTimeTer.h" // library to count the time elapsed.
#include "../../../../CenyML Library/otherLibraries/csv/csvManager.h" // library to open and create .csv files.
#include "../../../../CenyML Library/CenyML_Library/cpuSerial/statistics/CenyMLstatistics.h" // library to use the statistics methods from CenyML.


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
* then calculate the mean of each column. Finally, the results will be saved
* in a new .csv file for further comparations and validation purposes.
*
* @param int argc - This argument will posses the length number of what is
*		    contained within the argument "*argv[]".
*		    NOTE1: This argument will be at least "1" in value because
*		    its first argument is the title of this program.
*
* @param int *argv[] - This argument contains within the following:
*		       ARGUMENT1 = The directory of this program, including its name.
*		       ARGUMENT2 = All the characters you input on the terminal.
*
* @return 0
*
* @author Miranda Meza Cesar
* CREATION DATE: SEPTEMBER 23, 2021
* LAST UPDATE: OCTOBER 05, 2021
*/
int main(int argc, char** argv) {
	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	char csv1Directory[] = "../../../../Databases/statisticsDBs/multiplePolynomialEquationSystem_1000000samples_1Output_2Inputs.csv";
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file (which is declared in "csvManager.h").
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present or any of the rows contained in the target .csv file.
	
	// ---------------------- IMPORT DATA TO USE --------------------- //
	printf("Innitializing data extraction from .csv file ...\n");
	double startingTime, elapsedTime; // Declaration of variables used to count time in seconds.
	// Obtain the rows and columns dimensions of the data of the csv file (excluding headers)
	csv1.rowsAndColumnsDimensions = (double*)malloc( (double) (2 * sizeof(double)) ); // We initialize the variable that will store the rows & columns dimensions.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to obtain the data from the .csv file.
	getCsvRowsAndColumnsDimensions(&csv1); // We input the memory location of the "csv1" into the argument of this function to get the rows & columns dimensions.
	// From the structure variable "csv1", we allocate the memory required for the variable (csv1.allData) so that we can store the data of the .csv file in it.
	double n = csv1.rowsAndColumnsDimensions[0]; // total number of rows of the input matrix (X)
	double m = csv1.rowsAndColumnsDimensions[1]; // total number of columns of the input matrix (X)
	double totalElements = n * m;
	double nBytes = totalElements * sizeof(double);
	csv1.allData = (double*)malloc(nBytes);
	// Allocate the memory required for the struct variable "X", which will contain the input data of the system whose mean will be obtained.
	double* X = (double*)malloc(nBytes);
	// Store the csv data (excluding headers) in "X"
	getCsvFileData(&csv1); // We input the memory location of the "csv1" into the argument of this function to get all the data contained in the .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the data from the .csv file.
	printf("Data extraction from .csv file elapsed %f seconds.\n\n", elapsedTime);
	X = csv1.allData;
	
	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	
	// ------------------------ DATA SPLITTING ----------------------- //
	
	// ----------------------- FEATURE SCALING ----------------------- //
	
	// ------------------------ DATA SPLITTING ----------------------- //
	
	// ------------------------- DATA MODELING ----------------------- //
	printf("Innitializing CenyML mean method calculation ...\n");
	// Allocate the memory required for the variable "B_x_bar" (which will contain the mean of the input data "X") and innitialize it with zeros.
	totalElements = m;
	double* B_x_bar = (double*)calloc(totalElements, sizeof(double));
	// We calculate the mean for each of the columns available in the matrix "X" and the result is stored in the memory location of the pointer "B_x_bar".
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the mean of the data contain in the .csv file.
	getMean(X, n, m, B_x_bar);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the mean of the data contain in the .csv file.
	printf("CenyML mean method elapsed %f seconds.\n\n", elapsedTime);
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	// Define the desired file name and header names for the new .csv file to be create.
	char nameOfTheCsvFile[] = "CenyML_getMean_Results.csv"; // name the .csv file
	int currentRow;
    int currentColumn;
    char csvHeaders[strlen("id,system_id,dependent_variable,") + (int)(strlen("independent_variable_XXXXXXX")*(m-3))];
    csvHeaders[0] = '\0'; // Set a null value to this char variable.
	strcat(csvHeaders, "id,system_id,dependent_variable,"); // We add the first three column headers into "csvHeaders".
	char currentColumnInString[8]; // Variable used to store the string form of the currenColumn integer value.
    for (currentColumn = 3; currentColumn < (m-1); currentColumn++) { // We add the rest of the column headers into "csvHeaders"
    	strcat(csvHeaders, "independent_variable_");
    	sprintf(currentColumnInString, "%d", currentColumn);
    	strcat(csvHeaders, currentColumnInString);
    	strcat(csvHeaders, ",");
	} // We add the last header column.
	strcat(csvHeaders, "independent_variable_");
	sprintf(currentColumnInString, "%d", (int)(m-1));
	strcat(csvHeaders, currentColumnInString);
	// Create a new .csv file and save the results obtained in it.
	char isInsertId = 0; // Indicate through this flag variable that it is not desired that the file to be created automatically adds an "id" to each row.
	createCsvFile(nameOfTheCsvFile, csvHeaders, B_x_bar, 1, m, isInsertId); // We create the desired .csv file.
	
	
	return (0); // end of program.
}

