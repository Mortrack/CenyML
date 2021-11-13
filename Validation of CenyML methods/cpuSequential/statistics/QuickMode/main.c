/*
 * This program will read the .csv file "10systems_100samplesPerAxisPerSys.csv"
* to then exctact all its data and save it into the matrix "X". Subsequently,
* a new .csv file "CenyML_getQuickMode_Results.csv" will be created and in
* it, the quick mode values for each column will be saved for further
* comparations and validations of the "quick mode" method.
 */

 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
#include "../../../../CenyML library skeleton/otherLibraries/time/mTimeTer.h" // library to count the time elapsed.
#include "../../../../CenyML library skeleton/otherLibraries/csv/csvManager.h" // library to open and create .csv files.
#include "../../../../CenyML library skeleton/CenyML_Library/cpuSequential/statistics/CenyMLstatistics.h" // library to use the statistics methods of CenyML.


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
* then calculate the quick mode of all the data of that .csv file with respect
* to its columns. Finally, the results will be saved in a new .csv file for
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
* CREATION DATE: OCTOBER 19, 2021
* LAST UPDATE: NOVEMBER 08, 2021
*/
int main(int argc, char **argv) {
	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	char csv1Directory[] = "../../../../Databases/classificationDBs/randPolynomialClassificationSystem/10systems_100samplesPerAxisPerSys.csv"; // Directory of the reference .csv file
	char nameOfTheCsvFile[] = "CenyML_getQuickMode_Results.csv"; // Name the .csv file that will store the results.
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file (which is declared in "csvManager.h").
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	// NOTE: "desired_m" can be any value as long as (desired_m*n) <= 2'147'483'647, because of the long int max value (the compiler seems to activate the long data type when needed when using integer variables only).
	// desired_m <= 4'294 to comply with the note considering that n=100'000 and m=5.
	int desired_m = 5; // We define the desired number of columns that want to be processed with respect to the samples contained in the .csv file read by duplicating its columns.
	
	// ---------------------- IMPORT DATA TO USE --------------------- //
	printf("Innitializing data extraction from .csv file containing the reference input data ...\n");
	double startingTime, elapsedTime; // Declaration of variables used to count time in seconds.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to obtain the data from the reference .csv file.
	// Obtain the rows and columns dimensions of the data of the csv file (excluding headers)
	csv1.rowsAndColumnsDimensions = (int *) malloc(2*sizeof(int)); // We initialize the variable that will store the rows & columns dimensions.
	getCsvRowsAndColumnsDimensions(&csv1); // We input the memory location of the "csv1" into the argument of this function to get the rows & columns dimensions.
	// We save the rows and columns dimensions obtained in some variables that relate to the mathematical symbology according to the documentation of the method to be validated.
	int n = csv1.rowsAndColumnsDimensions[0]; // total number of rows of the input matrix (X)
	int m = csv1.rowsAndColumnsDimensions[1]; // total number of columns of the input matrix (X)
	// Before proceeding, we verify that the desired number of columns is an exact multiple of the rows contained in the reference .csv file.
	if ( (desired_m/m) != (desired_m/m) ) {
		printf("\nERROR: Variable \"desired_m\" must be a whole number and an exact multiple of the rows containted in the .csv file.\n");
		exit(1);
	}
	// From the structure variable "csv1", we allocate the memory required for the variable (csv1.allData) so that we can store the data of the .csv file in it.
	csv1.allData = (double *) malloc(n*m*sizeof(double));
	// We retrieve the data contained in the reference .csv file.
	getCsvFileData(&csv1); // We input the memory location of the "csv1" into the argument of this function to get all the data contained in the .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the data from the reference .csv file.
	printf("Data extraction from .csv file containing %d samples for each of the %d columns (total samples = %d), elapsed %f seconds.\n\n", n, m, (n*m), elapsedTime);
	
	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	printf("Innitializing input data with %d samples for each of the %d columns (total samples = %d)...\n", n, desired_m, (n*desired_m));
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to innitialize the input data to be used.
	// Allocate the memory required for the variable "X", which will contain the input data of the system to be evaluated.
	double *X = (double *) malloc(n*desired_m*sizeof(double));
	// Create the input data (X) with the same rows as in the reference .csv file and the desired number of columns by duplicating several times the data from such .csv file as needed.
	for (int currentRow=0; currentRow<n; currentRow++) {
		for (int currentColumn=0; currentColumn<(desired_m/m); currentColumn++) {
			for (int currentColumnCsv=0; currentColumnCsv<m; currentColumnCsv++) {
				X[(currentColumnCsv + currentColumn*m) + (currentRow*desired_m)] = csv1.allData[currentColumnCsv + currentRow*m];
			}
		}
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to innitialize the input data to be used.
	printf("Input data innitialization elapsed %f seconds.\n\n", elapsedTime);
	
	// ------------------------- DATA MODELING ----------------------- //
	printf("Innitializing CenyML Quick Mode method ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the current quick mode of the input data (X).
	// We allocate the memory of the following variables that will be required for the "quick mode" method.
	double *Mo = (double *) malloc(n*desired_m*sizeof(double)); // This variable will be used to store the data in which the "quick mode" method will be applied.
	int *Mo_n = (int *) calloc(desired_m, sizeof(int)); // This variable will be used to store the data in which the rows of each column of "Mo" will be specified.
	// We get the "quick mode" of the values contained in each of the columns available in the matrix "X".
	getQuickMode("quicksort", n, desired_m, X, Mo_n, Mo); // The result is stored in the memory location of the pointer "Mo".
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate all the current quick mode of the input data (X).
	printf("CenyML Quick Mode method elapsed %f seconds.\n\n", elapsedTime);
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .csv file which will store the results of the quick mode calculated.
	// Define the desired header names for the new .csv file to be create.
    char csvHeaders[strlen("id,system_id,dependent_variable,") + strlen("independent_variable_XXXXXXX")*(desired_m-3)]; // Variable where the following code will store the .csv headers.
    csvHeaders[0] = '\0'; // Innitialize this char variable with a null value.
	strcat(csvHeaders, "id,system_id,dependent_variable,"); // We add the first three column headers into "csvHeaders".
	char currentColumnInString[8]; // Variable used to store the string form of the currenColumn integer value, within the following for-loop.
    for (int currentColumn = 3; currentColumn < (desired_m-1); currentColumn++) { // We add the rest of the column headers into "csvHeaders"
    	strcat(csvHeaders, "independent_variable_");
    	sprintf(currentColumnInString, "%d", (currentColumn-2));
    	strcat(csvHeaders, currentColumnInString);
    	strcat(csvHeaders, ",");
	} // We add the last header column.
	strcat(csvHeaders, "independent_variable_");
	sprintf(currentColumnInString, "%d", (desired_m-3));
	strcat(csvHeaders, currentColumnInString);
	// Create a new .csv file and save the results obtained in it.
	char is_nArray = 1; // Indicate through this flag variable that the variable that indicates the samples (Mo_n) is an array because it has different samples per columns.
	char isInsertId = 0; // Indicate through this flag variable that it is not desired that the file to be created automatically adds an "id" to each row.
	createCsvFile(nameOfTheCsvFile, csvHeaders, Mo, Mo_n, is_nArray, desired_m, isInsertId); // We create the desired .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Creation of the .csv file to store the results obtained, elapsed %f seconds.\n\n", elapsedTime);
	printf("The program has been successfully completed!");
	
	
	// Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
	free(csv1.rowsAndColumnsDimensions);
	free(csv1.allData);
	free(X);
	free(Mo);
	free(Mo_n);
	return (0); // end of program.
}

