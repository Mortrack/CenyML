/*
 * Explanation of the purpose of this file.
 */

 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
//#include "otherLibraries/time/mTime.h" // library to count time during computer processing
//#include "otherLibraries/time/mTimeTer.h" // library to count time during computer processing
//#include "otherLibraries/cuda/check.h" // library to check if CUDA functions had an error.
#include "otherLibraries/csv/csvManager.h" // library to open and create .csv files
//#include "otherLibraries/pbPlots/pbPlots.h" // library to generate plots
//#include "otherLibraries/pbPlots/supportLib.h"  // library required for "pbPlots.h"


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
* This is the main function of the program. Here we will ...
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
* CREATION DATE: SEPTEMBER 22, 2021
* LAST UPDATE: N/A
*/

int main(int argc, char** argv) {
	// ---------- DECLARATION OF THE LOCAL VARIABLES TO USE ---------- //
	
	// ---------------------- IMPORT DATA TO USE --------------------- //
	/*
	// We open the desired file in read mode
	FILE *csvFile = fopen("newFile.csv", "r");
	// if the opned file was not available, then emit an error message. Otherwise, continue with the program
	if (csvFile == NULL) {
		perror("Unable to open de file"); // Emit a custom error message in the terminal. Afterwards, a detailed explanatory default message will be inserted next to your custom text.
		exit(1); // Force the termination of the program.
	}
	
	int maxCharPerRow = 65; // This variable is used so that the developer indicates the maximum number of characters that will be counted for any row read from the file that was opened.
    char line[maxCharPerRow]; // This variable is used in the process of obtaining the characters contained in the current line of the file that was opened.
    int currentRow = 1; // This variable will be used to know what is the current row read from the file that was opened (which will be from 1 up to max row)
    int currentColumn = 1; // This variable is used so that the developer can fix the desired column from which data is desired to be extrated, for each row read from the file that was opened.
    // The following while-loop is used to read all the rows and columns from the file that was opened.
	while (fgets(line, sizeof(line), csvFile))
    {
        char* charsFromCurrentLine = strdup(line); // This variable is used to get all the characters contained from the current line of the file that was opened.
        printf("Row: %d, Column: %d --> %s\n", currentRow, currentColumn, getfield(charsFromCurrentLine, currentColumn)); // Print in the data contained from the current row and current column in the terminal.
        free(charsFromCurrentLine); // Free the memory of the variable containing all the characters from the current row of the opened file.
        currentRow ++; // Increase the variable used as counter to know what the current row number is (which will be from 1 up to max row).
    }
    */
	
	// ------------- MATRIX INITIALIZATION THORUGH CPU --------------- //

	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	
	// ------------------ PRINT MATRIXES INFORMATION ----------------- //
	//printf("Matrixes size: rows= %d columns= %d\n", MATRIX_LENGTH, MATRIX_LENGTH);

	// ------------------------ DATA SPLITTING ----------------------- //
	
	// ----------------------- FEATURE SCALING ----------------------- //
	
	// ------------------------ DATA SPLITTING ----------------------- //
	
	// ------------------------- DATA MODELING ----------------------- //
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	/*
	// Trying the "pbPlots" library (https://github.com/InductiveComputerScience/pbPlots)
	double x [] = {-1,0,1,2,3,4,5,6};	
	double y [] = {-5,-4,-3,-2,1,0,1,2};
	// The sizeOf function will give 64 total bits because each inner
	// array value is 8bits. Therefore, divide between 8 to get the
	// total number of inner array components.
	int sizeOfX = sizeof(x)/8;
	int sizeOfY = sizeof(y)/8;
	RGBABitmapImageReference *imageReference = CreateRGBABitmapImageReference();
	DrawScatterPlot(imageReference, 600, 400, x, sizeOfX, y, sizeOfY);
	size_t length;
	double *pngData = ConvertToPNG(&length, imageReference->image);
	WriteToFile(pngData, length, "plot.png");
	*/
	
	/*
	// ----- CPU MEMORY ALLOCATION ----- //
	int matrixRows = 3;
	int matrixColumns = 3;
	int totalElementsPerMatrix = matrixRows * matrixColumns;
	double nBytes = totalElementsPerMatrix * sizeof(double);
	double *csvData = (double*)malloc(nBytes);
	
	// ----- MATRIX INNITIALIZATION ----- //
	// Define the data for the 1st row
	csvData[0] = 50; // (row1, column1)
	csvData[1] = 50; // (row1, column2)
	csvData[2] = 50; // (row1, column3)
	// Define the data for the 2nd row
	csvData[3] = 60; // (row2, column1)
	csvData[4] = 60; // (row2, column2)
	csvData[5] = 60; // (row2, column3)
	// Define the data for the 3rd row
	csvData[6] = 70; // (row3, column1)
	csvData[7] = 70; // (row3, column2)
	csvData[8] = 70; // (row3, column3)
	
	// ----- CSV FILE NAME AND HEADERS DEFINITION ----- //
	char nameOfTheCsvFile[15] = "newFile.csv"; // name the .csv file
	char csvHeaders[40] = "Student Id, Physics, Chemistry, Maths"; // indicate the desired headers for the .csv file
	
 	// ----- CREATE THE CSV FILE AND FILL IT WITH THE DESIRED DATA ----- //
	createCsvFile(nameOfTheCsvFile, csvHeaders, csvData, matrixRows, matrixColumns);
	*/
	
	
	return (0);
}


