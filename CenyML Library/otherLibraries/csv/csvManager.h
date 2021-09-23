#ifndef CSVMANAGER_H
#define CSVMANAGER_H

#include <string.h>
#include <stdio.h>

const char* getfield(char*, int);
void createCsvFile(char*, char*, double*, int, int);

#endif

