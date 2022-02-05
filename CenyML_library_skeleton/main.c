/*
 * Explain the purpose of this file in this section.
 */


 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
#include "otherLibraries/time/mTime.h" // library to count the time elapsed in Linux Ubuntu.
//#include "otherLibraries/time/mTimeTer.h" // library to count the time elapsed in Cygwin terminal window.
#include "otherLibraries/csv/csvManager.h" // library to open and create .csv files.
#include "otherLibraries/pbPlots/pbPlots.h" // library to generate plots v0.1.9.0
#include "otherLibraries/pbPlots/supportLib.h"  // library required for "pbPlots.h" v0.1.9.0
#include "CenyML_Library/cpuSequential/statistics/CenyMLstatistics.h" // library to use the statistics methods of CenyML.
#include "CenyML_Library/cpuSequential/featureScaling/CenyMLfeatureScaling.h" // library to use the feature scaling methods of CenyML.
#include "CenyML_Library/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.h" // library to use the regression evaluation metrics of CenyML.
#include "CenyML_Library/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.h" // library to use the classification evaluation metrics of CenyML.
#include "CenyML_Library/cpuSequential/machineLearning/CenyMLregression.h" // library to use the regression algorithms of CenyML.
#include "CenyML_Library/cpuSequential/machineLearning/CenyMLclassification.h" // library to use the classification algorithms of CenyML.
#include "CenyML_Library/cpuSequential/machineLearning/CenyMLdeepLearning.h" // library to use the deep learning algorithms of CenyML.
#include "CenyML_Library/cpuParallel/machineLearning/CenyMLdeepLearning_PC.h" // library to use the deep learning algorithms of CenyML with CPU parallelism.
#include "CenyML_Library/singleGpu/machineLearning/CenyMLdeepLearning_SG.h" // library to use the deep learning algorithms of CenyML with GPU parallelism.
#include "CenyML_Library/multiGpu/machineLearning/CenyMLdeepLearning_MG.h" // library to use the deep learning algorithms of CenyML with multi GPU parallelism.
#include "otherLibraries/cuda/CUDA_check.h" // library to use a define function to allow the tracking of CUDA errors in the terminal window.
#include <cuda_runtime.h>


// ---------------------------------------------- //
// ----- DEFINE THE GLOBAL RESOURCES TO USE ----- //
// ---------------------------------------------- //
// Write code here:


// --------------------------------------------------- //
// ----- The code from the main file starts here ----- //
// --------------------------------------------------- //
// Write code here:


// ----------------------------------------------- //
// ----- DEFINE THE GENERAL FUNCTIONS TO USE ----- //
// ----------------------------------------------- //
// Write code here:


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
* LAST UPDATE: FEBRUARY 2, 2022
*/

int main(int argc, char** argv) {
	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	// Write code here:
	
	
	// ---------------------- IMPORT DATA TO USE --------------------- //
	// Write code here:
	
	
	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	// Write code here:
	
	
	// ------------------------- DATA MODELING ----------------------- //
	// Write code here:
	
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	// Write code here:
	
	
	// NOTE: Remember to free the Heap memory used for the allocated
	// variables since they will no longer be used and then terminate the
	// program.
	return (0); // end of program.
}


