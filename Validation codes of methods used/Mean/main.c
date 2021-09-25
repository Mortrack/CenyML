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
	char csv1Directory[] = "../../Databases/regressionDBs/multiplePolynomialEquationSystem/multiplePolynomialEquationSystem_100systems_100samplesPerAxisPerSys.csv";
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file.
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present or any of the rows contained in the target .csv file.
	
	// ---------------------- IMPORT DATA TO USE --------------------- //
	// Obtain the rows and columns dimensions of the data of the csv file (excluding headers)
	csv1.rowsAndColumnsDimensions = (double*)malloc( (double) (2 * sizeof(double)) ); // We initialize the variable that will store the rows & columns dimensions.
	getCsvRowsAndColumnsDimensions(&csv1); // We input the memory location of the "csv1" into the argument of this function to get the rows & columns dimensions.
	// From the structure variable "csv1", we allocate the memory required for the variable (csv1.allData) that will retrieve the data of the .csv file.
	double n = csv1.rowsAndColumnsDimensions[0]; // total number of rows of the input matrix (X)
	double m = csv1.rowsAndColumnsDimensions[1]; // total number of columns of the input matrix (X)
	double totalElements = n * m;
	double nBytes = totalElements * sizeof(double);
	csv1.allData = (double*)malloc(nBytes);
	// Allocate the memory required for the struct variable "X", which will contain the input data of the system.
	double* X = (double*)malloc(nBytes);
	// Store the csv data (excluding headers) in "X"
	getCsvFileData(&csv1); // We input the memory location of the "csv1" into the argument of this function to get all the data contained in the .csv file.
	X = csv1.allData;
	
	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	
	// ------------------------ DATA SPLITTING ----------------------- //
	
	// ----------------------- FEATURE SCALING ----------------------- //
	
	// ------------------------ DATA SPLITTING ----------------------- //
	
	// ------------------------- DATA MODELING ----------------------- //
	// Allocate the memory required for the variable "B_x_bar", which will contain the mean of the input data "X".
	totalElements = m;
	nBytes = totalElements * sizeof(double);
	double* B_x_bar = (double*)malloc(nBytes);
	// We calculate the mean for each of the columns available in the matrix "X".
	getMean(X, n, m, B_x_bar);
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	// Display in the terminal the results obtained
    int currentRow;
    int currentColumn;
    for (currentColumn = 0; currentColumn < m; currentColumn++) {
		printf("Row: %d, Column: %d --> %f\n", currentRow, currentColumn, B_x_bar[currentColumn]);
	}
	// Define the desired file name and header names for the new .csv file to be create.
	char nameOfTheCsvFile[] = "CenyML_getMean_Results.csv"; // name the .csv file
	char csvHeaders[] = "id, system_id, dependent_variable, independent_variable_1, independent_variable_2"; // indicate the desired headers for the .csv file
	// Create a new .csv file and save the results obtained in it.
	createCsvFile(nameOfTheCsvFile, csvHeaders, B_x_bar, 1, m);
	
	
	// Free the allocated memory used and end this program.
	free(csv1.allData);
	free(csv1.rowsAndColumnsDimensions);
	free(X);
	free(B_x_bar);
	return (0);
}

