/*
 * This program will read the .csv file
* "multiplePolynomialEquationSystem_100systems_100samplesPerAxisPerSys.csv"
* to then exctact all its data and save it into the matrix "X". Subsequently,
* the mean of each row will be calculated and stored in "B_x_bar". Finally,
* a new .csv file "CenyML_getMean_Results.csv" will be created and in it, the
* means will be saved for further comparations and validations of the mean
* method.
 */

 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
#include "../../CenyML Library/otherLibraries/csv/csvManager.h" // library to open and create .csv files
#include "../../CenyML Library/CenyML_Library/cpuSerial/statistics/CenyMLstatistics.h" // library to use the statistics methods from CenyML


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
* LAST UPDATE: N/A
*/
int main(int argc, char** argv) {
	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	char csv1Directory[] = "../../Databases/regressionDBs/multiplePolynomialEquationSystem/";
	char csv1FileName[] = "multiplePolynomialEquationSystem_100systems_100samplesPerAxisPerSys.csv";
	double csv1MaxRowChars = 150;
	
	// ---------------------- IMPORT DATA TO USE --------------------- //
	double* csvDimensions = (double*)malloc( (double) (2 * sizeof(double)) );
	// Obtain the rows and columns dimensions of the data of the csv file (excluding headers)
	csvDimensions = getCsvFileDimensions(csv1Directory, csv1FileName, csv1MaxRowChars);
	double n = csvDimensions[0]; // total number of rows of the input matrix (X)
	double m = csvDimensions[1]; // total number of columns of the input matrix (X)
	// Allocate the memory required for the struct variable "X", which will contain the input data of the system.
	double totalElementsPerMatrix = n * m;
	double nBytes = totalElementsPerMatrix * sizeof(double);
	double* X = (double*)malloc(nBytes);
	// Store the csv data (excluding headers) in "X"
	X = getCsvFileData(csv1Directory, csv1FileName, csv1MaxRowChars, n, m);
	
	// ------------- MATRIX INITIALIZATION THORUGH CPU --------------- //

	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	
	// ------------------ PRINT MATRIXES INFORMATION ----------------- //
	
	// ------------------------ DATA SPLITTING ----------------------- //
	
	// ----------------------- FEATURE SCALING ----------------------- //
	
	// ------------------------ DATA SPLITTING ----------------------- //
	
	// ------------------------- DATA MODELING ----------------------- //
	// Allocate the memory required for the struct variable "X", which will contain the input data of the system.
	totalElementsPerMatrix = m;
	nBytes = totalElementsPerMatrix * sizeof(double);
	double* B_x_bar = (double*)malloc(nBytes);
	// We calculate the mean for each of the columns available in the matrix "X".
	B_x_bar = getMean(X, n, m);
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	// Display in the terminal the results obtained
    int currentRow;
    int currentColumn;
    for (currentColumn = 0; currentColumn < m; currentColumn++) {
		printf("Row: %d, Column: %d --> %f\n", currentRow, currentColumn, B_x_bar[currentColumn]);
	}
	// Define your desired file name of the new .csv file to create and of its headers.
	char nameOfTheCsvFile[30] = "CenyML_getMean_Results.csv"; // name the .csv file
	char csvHeaders[100] = "id, system_id, dependent_variable, independent_variable_1, independent_variable_2"; // indicate the desired headers for the .csv file
	// Create a new .csv file and save the results obtained in it.
	createCsvFile(nameOfTheCsvFile, csvHeaders, B_x_bar, 1, m);
	
	
	// Free the allocated memory used and end this program.
	free(X);
	free(B_x_bar);
	return (0);
}

