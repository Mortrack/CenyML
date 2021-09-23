#include "csvManager.h"

/**
* This is the main function of the program. Here we will read a .csv
* file and extract data from it to then print in the terminal. The
* purpose of this code is primarly to demonstrate how to extract data
* from a .csv file.
* 
* @param char line - This argument will possess the characters contained
*					 from a certain row of a determined file that was
*					 opened previouly with the "fopen()" function.
*
* @param int currentColumn - This argument contains within a number that
*							 indicates the desired column from which it
*							 is desired to extract the data, with respect
*							 to the line that was submitted through the
*							 argument variable "line".
* 
* @return string
* 
* @author Miranda Meza Cesar
* CREATION DATE: SEPTEMBER 13, 2021
* UPDATE DATE: N/A
*/
const char* getfield(char* line, int currentColumn) {
    const char* token;
    for (token = strtok(line, ",");
            token && *token;
            token = strtok(NULL, ",\n"))
    {
        if (!--currentColumn)
            return token;
    }
    return NULL;
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
* @param double data - This argument will contain the data of the
*					   matrix that wants to be written into the
*					   .csv file regarding the contents of the
*					   specified headers.
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
* CREATEION DATE: SEPTEMBER 22, 2021
* UPDATE DATE: N/A
*/
void createCsvFile(char* filename, char* header, double* data, int n, int m){
	// ----- CPU MEMORY ALLOCATION ----- //
	int currentRowAndCurrentColumn;
	
	// ----- CREATE CSV FILE AND WRITE THE HEADERS IN IT ----- //
	printf("\n Creating %s file",filename);
	FILE *fp;
	fp=fopen(filename,"w+");
	fprintf(fp, header);
	
	// ----- WRITE THE REQUESTED DATA INTO THE CSV FILE ----- //
	for(int currentRow=0; currentRow<n; currentRow++){
	    fprintf(fp,"\n%d",currentRow+1); // write the Id value of the current row
	    for(int currentColumn=0; currentColumn<m; currentColumn++) {
	    	currentRowAndCurrentColumn = currentColumn + currentRow * n;
			fprintf(fp,",%f ", data[currentRowAndCurrentColumn]);
		}
	}
	
	// ----- CLOSE THE CSV FILE ----- // 
	fclose(fp);
	printf("\n %sfile has been created",filename);
}

