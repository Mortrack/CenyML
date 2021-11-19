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
#ifndef CENYMLREGRESSION_H
#define CENYMLREGRESSION_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void getSimpleLinearRegression(double *, double *, int, int, int, double *);
void predictSimpleLinearRegression(double *, double *, int, int, int, double *);
void getMultipleLinearRegression(double *, double *, int, int, int, char, double *);
void predictMultipleLinearRegression(double *, double *, int, int, int, double *);
void getPolynomialRegression(double *, double *, int, int, int, int, char, double *);
void predictPolynomialRegression(double *, int, double *, int, int, int, double *);
void getMultiplePolynomialRegression(double *, double *, int, int, int, int, char, double *);
void predictMultiplePolynomialRegression(double *, int, double *, int, int, int, double *);

#endif

