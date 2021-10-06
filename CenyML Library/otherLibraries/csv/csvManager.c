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
* The function "getCsvRowsAndColumnsDimensions()" is used to obtain the
* dimensions of the rows and columns of a certain .csv file, excluding
* its headers.
* The results will be stored in the structure variable whose pointer is
* "csv" (csv->rowsAndColumnsDimensions[0] = rows dimension)
* (csv->rowsAndColumnsDimensions[1] = columns dimension).
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
* LAST UPDATE: OCTOBER 05, 2021
*/
void getCsvRowsAndColumnsDimensions(struct csvManager* csv) {
    // We initialize the required local variables for this function.
	int csvRowsLength = 0; // counter of the total number of rows of the .csv file.
	int csvColumnsLength = 0; // counter of the total number of columns of the .csv file.
	int maxRowCharacters = csv->maxRowChars * 40; // This variable is used to indicate the maximum number of characters that will be counted for any row read from the file that was opened.
    char line[maxRowCharacters]; // This variable is used in the process of obtaining the characters contained in the current line of the file that was opened.
    char* token; // This variable will be used to store the data of the current row vs the current column being read from the file.
    int currentRow = 0; // This variable will be used to know what is the current row read from the file.
    int currentColumn = 0; // This variable will be used to know what is the current column read from the file.
    
	// We open the desired file in read mode.
	FILE* csvFile = fopen(csv->fileDirectory, "r");
	
	// If the opned file was not available, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (csvFile == NULL) {
		printf("\nERROR: Unable to open the file. Either directory is too long or there is no such file or directory.\n");
		exit(1);
	}
	
	// We read the entire file to count how many rows and columns it has, for further memory allocation purposes in the sibling function "getCsvFileData()".
    while (fgets(line, sizeof(line), csvFile)) {		
        token = strtok(line, ",");
        while((token!=NULL) && (csvRowsLength==0)){
			token = strtok(NULL, ",");
			csvColumnsLength++;
		}
		csvRowsLength++;
    }
    csvRowsLength--;
    	
	// We save the obtained results.
	csv->rowsAndColumnsDimensions[0] = csvRowsLength; // We save the rows length in the index 0 of "csv->rowsAndColumnsDimensions".
	csv->rowsAndColumnsDimensions[1] = csvColumnsLength; // We save the columns length in the index 1 of "csv->rowsAndColumnsDimensions".
	
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
* LAST UPDATE: OCTOBER 05, 2021
*/
//TODO: Add an array type variable to the struct "csvManager" that saves the headers of the file.
void getCsvFileData(struct csvManager* csv) {
    // We initialize the required variables for this function.
	int csvRowsLength = csv->rowsAndColumnsDimensions[0]; // contains the total number of rows of the .csv file to open.
	int csvColumnsLength = csv->rowsAndColumnsDimensions[1]; // contains the total number of columns of the .csv file to open.
	int maxCharPerRow = csv->maxRowChars * 40; // This variable is used so that the developer indicates the maximum number of characters that will be counted for any row read from the file that was opened.
    char line[maxCharPerRow]; // This variable is used in the process of obtaining the characters contained in the current line of the file that was opened.
    char* token; // This variable will be used to store the data of the current row vs the current column being read from the file.
    int currentRow = 0; // This variable will be used to know what is the current row read from the file.
    int currentColumn = 0; // This variable will be used to know what is the current column read from the file.
    
	// We open the desired file in read mode.
	FILE* csvFile = fopen(csv->fileDirectory, "r");
	
	// If the opned file was not available, then emit an error message and terminate the program. Otherwise, continue with the program.
	if (csvFile == NULL) {
		printf("\nERROR: Unable to open the file. Either directory is too long or there is no such file or directory.\n");
		exit(1);
	}
	
    // We save the file data in the memory location address indicated by the pointer "csv->allData".
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
				csv->allData[currentColumn + (currentRow-1) * csvColumnsLength] = atof(token);
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
* @param char* filename - This argument is a pointer variable
*						 that will contain the string of the
*						 name that was requested for the .csv
*						 file to create.
*
* @param char* header - This argument is a pointer variable that
*					   will possess the string of the requested
*					   headers for the .csv file to create,
*					   which must be separated by a ",".
*
* @param double* data - This argument is the pointer to a memory
*					   allocated variable that has to contain the
*					   data of the matrix that wants to be
*					   written into the .csv file.
*
* @param int n - This argument will represent the total number
*				 of rows that the "data" variable argument will
*				 have.
*
* @param int m - This argument will represent the total number
*				 of columns that the "data" variable argument
*				 will have.
*
* @param char isInsertId = This argument variable will work as a
*						   flag to indicate with a "1" that it is
*						   desired that this function automatically
*						   adds an "id" value to each row or,
*						   conversely, with a "0" to indicate that
*						   it is not desired to automatically add
*						   these "id" values.
* 
* @return void
* 
* @author Miranda Meza Cesar
* CREATEION DATE: SEPTEMBER 23, 2021
* LAST UPDATE: OCTOBER 05, 2021
*/
void createCsvFile(char* filename, char* header, double* data, int n, int m, char isInsertId) {
	// Create the requested .csv file and write the specified headers in it.
	printf("Creating %s file ...",filename);
	FILE *fp;
	fp=fopen(filename,"w+");
	fprintf(fp, header);
	
	// Write the requested data into the .csv file.
	if (isInsertId == 1) { // If the flag "isInsertId"=1, then automatically add an "id" value to each row, along with the requested data, into the .csv file.
		for(int currentRow=0; currentRow<n; currentRow++){
		    fprintf(fp,"\n%f",currentRow+1); // write the Id value of the current row,
		    for(int currentColumn=0; currentColumn<m; currentColumn++) {
				fprintf(fp,",%f ", data[currentColumn + currentRow * m]);
			}
		}
	} else { // If the flag "isInsertId"!=1, then add the requested data into the .csv file with not "id" values.
		for(int currentRow=0; currentRow<n; currentRow++){
		    fprintf(fp,"\n%f", data[currentRow * m]); // write the first column value of the current row.
		    for(int currentColumn=1; currentColumn<m; currentColumn++) {
				fprintf(fp,",%f ", data[currentColumn + currentRow * m]);
			}
		}
	}
	
	// Close the .csv file.
	fclose(fp);
	printf("\n%sfile has been created\n",filename);
}

