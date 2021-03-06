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
#ifndef CSVMANAGER_H
#define CSVMANAGER_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

struct csvManager {
   char *fileDirectory; // Stores the directory path of the .csv file that wants to be managed.
   int maxRowChars; // Defines the expected maximum number of characters for any row contained in the .csv file of interest.
   double *allData; // Stores all the data contained in the .csv file of interest.
   int *rowsAndColumnsDimensions; // This variable must be innitialized as a 2-array, where the index 0 will contain the rows dimension (excluding headers) and index 1 the columns dimension of the .csv file data.
};

void getCsvRowsAndColumnsDimensions(struct csvManager *);
void getCsvFileData(struct csvManager *); //TODO: Add an array type variable to the struct "csvManager" that saves the headers of the file.
void createCsvFile(char *, char *, double *, int *, char, int, char);

#endif

