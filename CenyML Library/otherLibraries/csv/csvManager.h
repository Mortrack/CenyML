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

double* getCsvFileDimensions(char*, char*, double); //TODO: Find a way to unify the "getCsvFileDimensions" and the "getCsvFileData" functions into a single one.
double* getCsvFileData(char*, char*, double, double, double); //TODO: Add an array type variable to the struct "csvManager" that saves the headers of the file.
void createCsvFile(char*, char*, double*, double, double);

#endif

