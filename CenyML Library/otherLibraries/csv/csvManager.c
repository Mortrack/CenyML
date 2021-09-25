/*
   Copyright 2021 Cesar Miranda Meza (alias: Mortrack)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "csvManager.h"


/**
* The function "getCsvFileDimensions()" is used to obtain the dimensions
* of the rows and columns of a certain .csv file, excluding its headers.
* The results will be stored in the structure variable whose pointer is
* "csv" (csv->rowsAndColumnsDimensions[0] = rows dimension)
* (csv->rowsAndColumnsDimensions[1] = columns dimension)
* 
* @param struct csvManager* csv - This argument is defined as a pointer
*								  of a structure variable that is used
*								  to store the parameter required for
*								  the management of a certain .csv file
*								  (for more details see the header file).
* 
* @return void
* 
* @author Cesar Miranda Meza
* CREATION DATE: SEPTEMBER 23, 2021
* UPDATE DATE: SEPTEMBER 25, 2021
*/
void getCsvRowsAndColumnsDimensions(struct csvManager* csv) {
    // We initialize the required variables for this function.
	double csvRowsLength = 0; // counter of the total number of rows of the .csv file.
	double csvColumnsLength = 0; // counter of the total number of columns of the .csv file.
	int maxRowCharacters = csv->maxRowChars * 40; // This variable is used so that the developer indicates the maximum number of characters that will be counted for any row read from the file that was opened.
    char line[maxRowCharacters]; // This variable is used in the process of obtaining the characters contained in the current line of the file that was opened.
    char* token; // This variable will be used to store the data of the current row vs the current column being read from the file.
    int currentRow = 0; // This variable will be used to know what is the current row read from the file.
    int currentColumn = 0; // This variable will be used to know what is the current column read from the file.
    
	// We open the desired file in read mode.
	FILE* csvFile = fopen(csv->fileDirectory, "r");
	
	// if the opned file was not available, then emit an error message. Otherwise, continue with the program.
	if (csvFile == NULL) {
		printf("\nERROR: Unable to open the file. Either directory is too long or there is no such file or directory.\n"); // Emit a custom error message in the terminal. Afterwards, a detailed explanatory default message will be inserted next to your custom text.
		exit(1); // Force the termination of the program.
	}
	
	// We read the entire file to count how many rows and columns it has, for further memory allocation purposes in the function "getCsvFileData()".
    while (fgets(line, sizeof(line), csvFile)) {		
        token = strtok(line, ",");
        while((token!=NULL) && (csvRowsLength==0)){
			token = strtok(NULL, ",");
			csvColumnsLength++;
		}
		csvRowsLength++;
    }
    csvRowsLength--;
    	
	// We allocate the memory required for the return variable that will store the dimensions of the .csv file.
	csv->rowsAndColumnsDimensions[0] = csvRowsLength;
	csv->rowsAndColumnsDimensions[1] = csvColumnsLength;
	
	// We close the .csv file.
	fclose(csvFile);
}


/**
* The function "getCsvFileData()" is used to obtain all the data contained
* within the desired .csv file. All variable parameters required for the
* management of such file are stored in the structure variable csvManager
* whose pointer is "csv". It is in this same structre variable where the
* extracted data will be stored (csv->allData).
* 
* @param struct csvManager* csv - This argument is defined as a pointer
*								  of a structure variable that is used
*								  to store the parameter required for
*								  the management of a certain .csv file
*								  (for more details see the header file).
* 
* @return void
* 
* @author Cesar Miranda Meza
* CREATION DATE: SEPTEMBER 23, 2021
* UPDATE DATE: SEPTEMBER 25, 2021
*/
//TODO: Add an array type variable to the struct "csvManager" that saves the headers of the file.
void getCsvFileData(struct csvManager* csv) {
    // We initialize the required variables for this function.
	double csvRowsLength = csv->rowsAndColumnsDimensions[0]; // contains the total number of rows of the .csv file to open.
	double csvColumnsLength = csv->rowsAndColumnsDimensions[1]; // contains the total number of columns of the .csv file to open.
	int maxCharPerRow = csv->maxRowChars * 40; // This variable is used so that the developer indicates the maximum number of characters that will be counted for any row read from the file that was opened.
    char line[maxCharPerRow]; // This variable is used in the process of obtaining the characters contained in the current line of the file that was opened.
    char* token; // This variable will be used to store the data of the current row vs the current column being read from the file.
    int currentRow = 0; // This variable will be used to know what is the current row read from the file.
    int currentColumn = 0; // This variable will be used to know what is the current column read from the file.
    
	// We open the desired file in read mode.
	FILE* csvFile = fopen(csv->fileDirectory, "r");
	
	// if the opned file was not available, then emit an error message. Otherwise, continue with the program.
	if (csvFile == NULL) {
		printf("\nERROR: Unable to open the file. Either directory is too long or there is no such file or directory.\n"); // Emit a custom error message in the terminal. Afterwards, a detailed explanatory default message will be inserted next to your custom text.
		exit(1); // Force the termination of the program.
	}
	
    // We save the file data in the variable indicated by the pointer "csv->allData".
    while (fgets(line, sizeof(line), csvFile)) {
    	// We discard reading the data of the row containing the header of the file.
        if (currentRow == 0) {
        	token = strtok(line, ",");
        	currentColumn = 0;
        	while(token != NULL){
				token = strtok(NULL, ",");
				currentColumn++;
        	}
		}
		currentRow++;
		// We start saving the data contained in the file.
		while (fgets(line, sizeof(line), csvFile)) {
			token = strtok(line, ",");
	        currentColumn = 0;
	        while(token != NULL){       	
				csv->allData[currentColumn + (currentRow-1) * (int) csvColumnsLength] = atof(token);
				token = strtok(NULL, ",");
				currentColumn++;
			}
			currentRow++;
		}
    }
    
    // We close the .csv file.
	fclose(csvFile);
}


/**
* The "createCsvFile()" function is used to create a comma
* delimmited .csv file and write specified headers and data
* into it.
* 
* @param char filename - This argument will contain the string
*						 of the name that was requested for the
*						 .csv file to create.
*
* @param char header - This argument will possess the string of
*					   the requested headers for the .csv file
*					   to create, which are separated by a ",".
*
* @param double data - This argument is a memory allocated
*					   variable that will contain the data of
*					   the matrix that wants to be written into
*					   the .csv file.
*
* @param int n - This argument will represent the total number
*				 of rows that the "data" variable argument will
*				 have.
*
* @param int m - This argument will represent the total number
*				 of columns that the "data" variable argument
*				 will have.
* 
* @return void
* 
* @author Miranda Meza Cesar
* CREATEION DATE: SEPTEMBER 23, 2021
* UPDATE DATE: N/A
*/
void createCsvFile(char* filename, char* header, double* data, double n, double m) {
	// Create the requested .csv file and write the specified headers in it.
	printf("\n Creating %s file",filename);
	FILE *fp;
	fp=fopen(filename,"w+");
	fprintf(fp, header);
	
	// Write the requested data into the .csv file.
	for(double currentRow=0; currentRow<n; currentRow++){
	    fprintf(fp,"\n%f",currentRow+1); // write the Id value of the current row
	    for(double currentColumn=0; currentColumn<m; currentColumn++) {
			fprintf(fp,",%f ", data[(int)(currentColumn + currentRow * m)]);
		}
	}
	
	// Close the .csv file.
	fclose(fp);
	printf("\n %sfile has been created",filename);
}

