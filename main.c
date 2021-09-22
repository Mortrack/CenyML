/*
 * Explanation of the purpose of this file.
 */

 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
//#include "otherLibraries/time/mTime.h" // library to count time during computer processing
#include "otherLibraries/time/mTimeTer.h" // library to count time during computer processing
//#include "otherLibraries/cuda/check.h" // library to check if CUDA functions had an error.
#include "otherLibraries/pbPlots/pbPlots.h" // library to generate plots
#include "otherLibraries/pbPlots/supportLib.h"  // library required for "pbPlots.h"

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

	// -------------------- CPU MEMORY ALLOCATION -------------------- //
	
	// ------------- MATRIX INITIALIZATION THORUGH CPU --------------- //

	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	
	// ------------------ PRINT MATRIXES INFORMATION ----------------- //
	//printf("Matrixes size: rows= %d columns= %d\n", MATRIX_LENGTH, MATRIX_LENGTH);

	// ------------------------ DATA SPLITTING ----------------------- //
	
	// ----------------------- FEATURE SCALING ----------------------- //
	
	// ------------------------ DATA SPLITTING ----------------------- //
	
	// ------------------------- DATA MODELING ----------------------- //
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
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
	

	return (0);
}


