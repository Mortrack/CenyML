/*
 * This program will read the .csv file
* "multiplePolynomialEquationSystem_100systems_100samplesPerAxisPerSys.csv"
* to then exctact all its data and save it into the matrix "X". Subsequently,
* the sorted vales of eacy column and each possible permutation will be
* calculated and only one of its results will be stored in "newX" (because all
* of the results will be identical). Finally, a new .csv file
* "CenyML_getQuickSort_Results.csv" will be created and in it, the sorted
* values for each column will be saved for further comparations and
* validations of the "quick sort" method.
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
/**
* The "getPermutations()" function is inspired in the code made and
* explained by "GeeksforGeeks" in the Youtube video "Write a program
* to print all permutations of a given string | GeeksforGeeks"
* (URL = https://www.youtube.com/watch?v=AfxHGNRtFac). Nonetheless,
* as coded in this file, this function is used to obtain all the
* possible permutations of the "m" elements contained in the first
* row of the pointer argument variable "inputMatrix". Finally, the
* results obtained will be stored in "inputMatrix" from its second
* row up to the !m-th ("!m" = factorial of m) row.
*
* @param int l - This argument will be used to determine one of
*				 the two values that will have to be swaped for
*				 the current permutation to be applied. When
*				 calling this function, it will be necessary to
*				 input this argument variable with the value of 0.
*
* @param int r - This argument will be used to determine one of
*				 the two values that will have to be swaped for
*				 the current permutation to be applied. When
*				 calling this function, it will be necessary to
*				 input this argument variable with the value of (m-1).
*
* @param int m - This argument will represent the total number
*				 of columns that the "inputMatrix" variable argument
*				 will have.
*
* @param int *currentRow - This argument will contain the pointer to
*						a memory allocated variable that will be used
*						to know in which row, of the argument pointer
*						variable "inputMatrix", to store the result of
*						the currently identified permutation. This is
*						necessary because this function will be solved
*						recursively and each time this function is
*						is called, a different permutation possition
*						will be obtained. NOTE THAT IT IS
*						INDISPENSABLE THAT THIS VARIABLE IS ALLOCATED
*						BEFORE CALLING THIS FUNCTION AND INNITIALIZED
*						WITH A VALUE OF 0.
*
* @param double *inputMatrix - This argument will contain the
*							   pointer to a memory allocated
*							   input matrix in which all the
*							   identified permutations with
*							   respect to the "m" elements of its first
*							   row, will be stored. These resulting
*							   permutation positions will be stored
*							   from the 2nd row up to the !m-th row
*							   ("!m" = factorial of m). NOTE THAT THIS
*							   VARIABLE MUST BE PREVIOUSLY ALLOCATED.
*							   FURTHERMORE, THE DATA TO BE PERMUTATED
*							   IS THE ONE THAT WAS STORED IN EACH
*							   COLUMN OF THE FIRST ROW OF THIS
*							   VARIABLE BEFORE CALLING THIS FUNCTION.
*							   FINNALLY, REMEMBER THAT THE TOTAL NUMBER
*							   OF PERMUTATIONS TO BE STORED WILL BE
*							   EQUAL TO "!m" AND THEREFORE, THIS
*							   VARIABLE WILL REQUIRE TO HAVE "m"
*							   COLUMNS AND "!m" ROWS ALLOCATED IN IT.
*
* NOTE: RESULT IS STORED IN THE MEMORY ALLOCATED POINTER VARIABLE
*       "inputMatrix". THERE, YOU WILL FIND ALL THE POSSIBLE
*		COMBINATION PERMUTATIONS WITH RESPECT TO THE DATA STORED IN THE
*		FIRST ROW OF "inputMatrix", EACH ONE IN A SEPERATED ROW.
*
* 
* @return void
*
* @author Miranda Meza Cesar
* CREATION DATE: OCTOBER 17, 2021
* LAST UPDATE: NOVEMBER 02, 2021
*/
void getPermutations(int l, int r, int m, int *currentRow, int *inputMatrix) {
    int i;
    int dataHolder;
    if (l == r) {
    	// Store the currently identified permutation.
        for (int currentColumn=0; currentColumn<m; currentColumn++) {
        	inputMatrix[currentColumn + (*currentRow)*m] = inputMatrix[currentColumn];
		}
		
		// Indicate to the next recursive function to store the next permutation in the next row of the pointer variable "inputMatrix".
		*currentRow = *currentRow + 1;
    }
    else {
        for (i = l; i <= r; i++) {
            // swap data
            dataHolder = inputMatrix[l];
            inputMatrix[l] = inputMatrix[i];
            inputMatrix[i] = dataHolder;
            
            // Apply recursive permutation
            getPermutations(l+1, r, m, currentRow, inputMatrix);
            
            // make a backtrack swap
            dataHolder = inputMatrix[l];
            inputMatrix[l] = inputMatrix[i];
            inputMatrix[i] = dataHolder;
        }
    }
}


// ----------------------------------------- //
// ----- THE MAIN FUNCTION STARTS HERE ----- //
// ----------------------------------------- //
/**
* This is the main function of the program. Here we will read a .csv file and
* then calculate the sort of each column by applying the "quick sort" method.
* This method will be applied for each possible permutation for each column of
* the input data matrix. Finally, the results will be saved in a new .csv file
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
* CREATION DATE: OCTOBER 07, 2021
* LAST UPDATE: NOVEMBER 02, 2021
*/
int main(int argc, char **argv) {
	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	char csv1Directory[] = "../../../../Databases/regressionDBs/randGaussianEquationSystem/gaussianEquationSystem_1systems_10samplesPerSys.csv"; // Directory of the reference .csv file
	char nameOfTheCsvFile[] = "CenyML_getQuickSort_Results.csv"; // Name the .csv file that will store the results.
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file (which is declared in "csvManager.h").
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	// NOTE: "desired_m" can be any value as long as (desired_m*n) <= 2'147'483'647, because of the long int max value (the compiler seems to activate the long data type when needed when using integer variables only).
	// desired_m <= 53'687'091 to comply with the note considering that n=10 and m=4.
	int desired_m = 1000; // We define the desired number of columns that want to be processed with respect to the samples contained in the .csv file read by duplicating its columns.
	int permutationIndexToEvaluate = 0; // We define the index of the permutation that we specifically want to evaluate.
	
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
	// Allocate the memory required for the variable "X", which will contain the input data of the system to be evaluated.
	double *X = (double *) malloc(n*desired_m*sizeof(double));
	// We retrieve the data contained in the reference .csv file.
	getCsvFileData(&csv1); // We input the memory location of the "csv1" into the argument of this function to get all the data contained in the .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the data from the reference .csv file.
	printf("Data extraction from .csv file containing %d samples for each of the %d columns (total samples = %d), elapsed %f seconds.\n\n", n, m, (n*m), elapsedTime);
	
	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	// Create the input data (X) with the same rows as in the reference .csv file and the desired number of columns by duplicating several times the data from such .csv file as needed.
	printf("Innitializing input data with %d samples for each of the %d columns (total samples = %d)...\n", n, desired_m, (n*desired_m));
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to innitialize the input data to be used.
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
	printf("Innitializing CenyML Quick Sort method ...\n");
	// Obtain the number of possible permutations (which will be the factorial of "n").
	int factorialValue = n;
	for (int i=1; i<n; i++) {
		factorialValue = factorialValue*i;
	}
	// Innitialize the data of the first row of the pointer variable to be permutated with the values of all the possible indexes that the matrix "X" can have (which is "0" up to "n").
	int *indexIndicator = (int *) malloc(factorialValue*n*sizeof(int));
	for (int currentColumn=0; currentColumn<n; currentColumn++) {
		indexIndicator[currentColumn] = currentColumn;
	}
	// Get all possible index permutations that were stored in "indexIndicator".
	int *currentPermutation = (int *) calloc(1, sizeof(int));
	getPermutations(0, (n-1), n, currentPermutation, indexIndicator);
	// Excecute the "quick sort" method over each permutation identified.
	elapsedTime = 0;
	double *newX = (double *) malloc(n*desired_m*sizeof(double)); // This variable will be used to store the data in which the "quick sort" method will be applied.
	for (int currentPermutation=0; currentPermutation<factorialValue; currentPermutation++) {
		// Fill "newX" with the data of X but arranged with the current permutation indicated in "indexIndicator".
		for (int currentRow=0; currentRow<n; currentRow++) {
			for (int currentColumn=0; currentColumn<desired_m; currentColumn++) {
		        newX[currentColumn + currentRow*desired_m] = X[currentColumn + indexIndicator[currentRow + currentPermutation*n]*desired_m];
		    }
		}
		// We sort the values contained in each of the columns available in the matrix "newX" and the result is stored in the memory location of the pointer "newX".
		if (currentPermutation == permutationIndexToEvaluate) {
			startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the current sort of the input data (X).
			getSort("quicksort", n, desired_m, newX);
			elapsedTime = elapsedTime + (seconds() - startingTime); // We obtain the elapsed time to calculate all the current sorts of the input data (X).
			break;
		}
	}
	printf("CenyML Quick Sort method elapsed %f seconds.\n\n", elapsedTime);
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .csv file which will store the results of the sort calculated.
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
	char is_nArray = 0; // Indicate through this flag variable that the variable that indicates the samples (n) is not an array because it has the same amount of samples per columns.
	char isInsertId = 0; // Indicate through this flag variable that it is not desired that the file to be created automatically adds an "id" to each row.
	createCsvFile(nameOfTheCsvFile, csvHeaders, newX, &n, is_nArray, desired_m, isInsertId); // We create the desired .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Creation of the .csv file to store the results obtained, elapsed %f seconds.\n\n", elapsedTime);
	printf("The program has been successfully completed!");
	
	
	// Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
	free(csv1.rowsAndColumnsDimensions);
	free(csv1.allData);
	free(X);
	free(currentPermutation);
	free(indexIndicator);
	free(newX);
	return (0); // end of program.
}

