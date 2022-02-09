/*
 * This program will read the .csv file "100systems_100samplesPerAxisPerSys.csv"
* to then exctact all its data and save it into the matrix "X". Subsequently,
* the mean intervals of each row will be calculated and stored in
* "meanIntervals". Finally, a new .csv file
* "CenyML_get99_9perCentMeanIntervals_Results.csv" will be created and in it, the
* means for each column will be validated with the results obtained with the
* Excel's "Data analysis tool" which will be extracted from the .csv file
* "99_9perCentMeanIntervals_ExcelResults". Finally, the obtained mean intervals
* with the CenyML library will be saved in a .csv file for additional inspection.
 */

 // ------------------------------------------------- //
 // ----- DEFINE THE LIBRARIES THAT WE WILL USE ----- //
 // ------------------------------------------------- //
#include <stdio.h>
#include <stdlib.h>
#include "../../../../CenyML_library_skeleton/otherLibraries/time/mTime.h" // library to count the time elapsed in Linux Ubuntu.
//#include "../../../../CenyML_library_skeleton/otherLibraries/time/mTimeTer.h" // library to count the time elapsed in Cygwin terminal window.
#include "../../../../CenyML_library_skeleton/otherLibraries/csv/csvManager.h" // library to open and create .csv files.
#include "../../../../CenyML_library_skeleton/CenyML_Library/cpuSequential/statistics/CenyMLstatistics.h" // library to use the statistics methods of CenyML.


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
* This is the main function of the program. Here we will read two .csv files.
* One .csv file will contain the input data and the other the comparison data.
* Then, the mean intervals of each column from the input .csv file will be
* calculated. Subsequently, the obtained mean intervals will be validated with
* respect to the comparison data. Finally, the results will be saved in a new
* .csv file for further inspection purposes.
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
* CREATION DATE: JANUARY 11, 2021
* LAST UPDATE: N/A
*/
int main(int argc, char **argv) {
	// --- LOCAL VARIABLES VALUES TO BE DEFINED BY THE IMPLEMENTER --- //
	char csv1Directory[] = "../../../../databases/regression/randMultiplePolynomialEquationSystem/100systems_100samplesPerAxisPerSys.csv"; // Directory of the reference .csv file to be used by the CenyML algorithm.
	char csv2Directory[] = "99_9perCentMeanIntervals_ExcelResults.csv"; // Directory of the reference .csv file that contains the results of Excel that were obtained using its "Data Analysis" tool, for comparative purposes.
	char nameOfTheCsvFile[] = "CenyML_get99_9perCentMeanIntervals_Results.csv"; // Name the .csv file that will store the results.
	struct csvManager csv1; // We create a csvManager structure variable to manage the desired .csv file (which is declared in "csvManager.h").
	csv1.fileDirectory = csv1Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv1.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	struct csvManager csv2; // We create a csvManager structure variable to manage the desired .csv file (which is declared in "csvManager.h").
	csv2.fileDirectory = csv2Directory; // We save the directory path of the desired .csv file into the csvManager structure variable.
	csv2.maxRowChars = 150; // We define the expected maximum number of characters the can be present for any of the rows contained in the target .csv file.
	int degreesOfFreedom = 0; // Desired degrees of freedom to be applied in the standard deviation to be calculated. A "0" would represent a degrees of freedom of "n", a "1" would represent a "n-1", ..., a "degrees" would represent a "n-degrees".
	char isTdistribution = 0; // Flag variable used to indicate if the area under the curve to be calculated for the mean intervals will be obtained with the t distribution (int 1) or with the z distribution (int 0).
	float desiredTrustInterval = 99.9; // We define the numeric porcentage that is desired for the trust interval to be used to calculate the mean intervals.
	
	
	// ---------------------- IMPORT DATA TO USE --------------------- //
	printf("Initializing data extraction from csv files containing the input data to be used for the CenyML algorithm and the csv containing the Excel results for comparison purposes ...\n");
	double startingTime, elapsedTime; // Declaration of variables used to count time in seconds.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to obtain the data from the reference .csv files.
	// GET THE INPUT DATA TO BE USED FOR THE CENYML ALGORITHM
	// Obtain the rows and columns dimensions of the data of the csv1 file (excluding headers)
	csv1.rowsAndColumnsDimensions = (int *) malloc(2*sizeof(int)); // We initialize the variable that will store the rows & columns dimensions.
	getCsvRowsAndColumnsDimensions(&csv1); // We input the memory location of the "csv1" into the argument of this function to get the rows & columns dimensions.
	// We save the rows and columns dimensions obtained in some variables that relate to the mathematical symbology according to the documentation of the method to be validated.
	int n = csv1.rowsAndColumnsDimensions[0]; // total number of rows of the input matrix (X)
	int m = csv1.rowsAndColumnsDimensions[1]; // total number of columns of the input matrix (X)
	// From the structure variable "csv1", we allocate the memory required for the variable (csv1.allData) so that we can store the data of the .csv file in it.
	csv1.allData = (double *) malloc(n*m*sizeof(double));
	// We retrieve the data contained in the csv1 file.
	getCsvFileData(&csv1); // We input the memory location of the "csv1" into the argument of this function to get all the data contained in the .csv file.
	
	// GET THE RESULTS THAT WERE OBTAINED WITH EXCEL THROUGH ITS "DATA ANALYSIS" TOOL.
	// Obtain the rows and columns dimensions of the data of the csv2 file (excluding headers)
	csv2.rowsAndColumnsDimensions = (int *) malloc(2*sizeof(int)); // We initialize the variable that will store the rows & columns dimensions.
	getCsvRowsAndColumnsDimensions(&csv2); // We input the memory location of the "csv2" into the argument of this function to get the rows & columns dimensions.
	// We save the rows and columns dimensions obtained in some variables that relate to the mathematical symbology according to the documentation of the method to be validated.
	int csv2Rows = csv2.rowsAndColumnsDimensions[0]; // total number of rows of the csv2 file.
	int csv2Columns = csv2.rowsAndColumnsDimensions[1]; // total number of columns of the csv2 file.
	// From the structure variable "csv2", we allocate the memory required for the variable (csv2.allData) so that we can store the data of the .csv file in it.
	csv2.allData = (double *) malloc(csv2Rows*csv2Columns*sizeof(double));
	// We retrieve the data contained in the csv2 file.
	getCsvFileData(&csv2); // We input the memory location of the "csv2" into the argument of this function to get all the data contained in the .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to obtain the data from the reference .csv file.
	printf("Data extraction from the csv files elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------------ PREPROCESSING OF THE DATA ------------------ //
	printf("Initializing the comparison data to be used (from Excel) and the input data with %d samples for each of the %d columns (total samples = %d)...\n", n, m, (n*m));
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to innitialize the input data to be used.
	// Allocate the memory required for the variable "X", which will contain the input data of the system under study.
	double *X = (double *) malloc(n*m*sizeof(double));
	// Pass the extracted input data from the .csv file to the variable "X".
	for (int currentRow=0; currentRow<n; currentRow++) {
		for (int currentColumn=0; currentColumn<m; currentColumn++) {
			X[currentColumn + currentRow*m] = csv1.allData[currentColumn + currentRow*m];
		}
	}
	// Allocate the memory required for the variable "excelResults", which will contain the Excel results that were obtained and that will be used for comparison purposes.
	double *excelResults = (double *) malloc(csv2Rows*csv2Columns*sizeof(double));
	// Pass the extracted Excel results data from the .csv file to the variable "excelResults".
	for (int currentRow=0; currentRow<csv2Rows; currentRow++) {
		for (int currentColumn=0; currentColumn<csv2Columns; currentColumn++) {
			excelResults[currentColumn + currentRow*csv2Columns] = csv2.allData[currentColumn + currentRow*csv2Columns];
		}
	}
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to innitialize the input data to be used.
	printf("Comparison and input data innitialization elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------------------- DATA MODELING ----------------------- //
	printf("Initializing CenyML mean interval method calculation ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to calculate the mean of the input data (X).
	// Allocate the memory required for the variable "mean" (which will contain the mean of the input data "X") and innitialize it with zeros.
	double *mean = (double *) calloc(m, sizeof(double));
	// We calculate the mean for each of the columns available in the matrix "X" and the result is stored in the memory location of the pointer "mean".
	getMean(X, n, m, mean);
	// Allocate the memory required for the variable "standardDeviation" (which will contain the standard deviation of the input data "X") and innitialize it with zeros.
	double *standardDeviation = (double *) calloc(m, sizeof(double));
	// We calculate the standard deviation for each of the columns available in the matrix "X" and the result is stored in the memory location of the pointer "standardDeviation".
	getStandardDeviation(X, n, m, degreesOfFreedom, standardDeviation);
	// Allocate the memory required for the variable "meanIntervals" (which will contain the mean intervals to be calculated).
	double *meanIntervals = (double *) malloc(m*2*sizeof(double));
	// We calculate the mean intervals for each of the individually calculated means and standard deviations.
	getMeanIntervals(mean, standardDeviation, desiredTrustInterval, isTdistribution, n, m, meanIntervals);
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to calculate the mean of the input data (X).
	printf("CenyML mean interval method elapsed %f seconds.\n\n", elapsedTime);
	
	
	// ------------ PREDICTIONS/VISUALIZATION OF THE MODEL ----------- //
	printf("Initializing validation of the CenyML mean interval method ...\n");
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to validate the mean interval method.
	// We validate the mean interval method.
	double differentiation; // Variable used to store the error obtained for a certain value.
	double epsilon = 1.6E-1; // Variable used to store the max error value permitted during validation process.
	char isMatch = 1; // Variable used as a flag to indicate if the current comparation of values stands for a match. Note that the value of 1 = is a match and 0 = is not a match.
	// We check that all the differentiations do not surpass the error indicated through the variable "epsilon".
	for (int currentRow=0; currentRow<m; currentRow++) {
		for (int currentColumn=0; currentColumn<2; currentColumn++) {
			differentiation = fabs(meanIntervals[currentColumn + currentRow*2] - excelResults[(currentColumn+1) + currentRow*3]);
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
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to validate the mean interval method.
	printf("The validation of the CenyML mean interval method elapsed %f seconds.\n\n", elapsedTime);
	
	// We save the results obtained for the mean interval method.
	startingTime = seconds(); // We obtain the reference time to count the elapsed time to create the .csv file which will store the results that were obtained.
	// Define the desired header names for the new .csv file to be create.
    char csvHeaders[strlen("lower_mean_interval,upper_mean_interval") + 1]; // Variable where the following code will store the .csv headers.
    csvHeaders[0] = '\0'; // Innitialize this char variable with a null value.
	strcat(csvHeaders, "lower_mean_interval,upper_mean_interval"); // We add desired the column headers into "csvHeaders".
	// Create a new .csv file and save the results obtained in it.
	char is_nArray = 0; // Indicate through this flag variable that the variable that indicates the samples (1) is not an array because it has the same amount of samples per columns.
	char isInsertId = 0; // Indicate through this flag variable that it is not desired that the file to be created automatically adds an "id" to each row.
	int csvFile_n = m; // This variable is used to indicate the number of rows with data that will be printed in the .csv file to be created.
	createCsvFile(nameOfTheCsvFile, csvHeaders, meanIntervals, &csvFile_n, is_nArray, 2, isInsertId); // We create the desired .csv file.
	elapsedTime = seconds() - startingTime; // We obtain the elapsed time to create the .csv file which will store the results calculated.
	printf("Creation of the .csv file to store the results obtained, elapsed %f seconds.\n\n", elapsedTime);
	printf("The program has been successfully completed!\n");
	
	
	// Free the Heap memory used for the allocated variables since they will no longer be used and then terminate the program.
	free(csv1.rowsAndColumnsDimensions);
	free(csv1.allData);
	free(csv2.rowsAndColumnsDimensions);
	free(csv2.allData);
	free(X);
	free(excelResults);
	free(mean);
	free(standardDeviation);
	free(meanIntervals);
	return (0); // end of program.
}

