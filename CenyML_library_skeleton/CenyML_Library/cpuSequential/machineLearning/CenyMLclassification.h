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
#ifndef CENYMLCLASSIFICATION_H
#define CENYMLCLASSIFICATION_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void getLinearLogisticClassification(double *, double *, int, int, int, double, char, double *);
void predictLinearLogisticClassification(double *, double, double *, int, int, int, double *);
void getSimpleLinearMachineClassification(double *, double *, int, int, int, char, double *);
void predictSimpleLinearMachineClassification(double *, double *, int, int, int, double *);
void getKernelMachineClassification(double *, double *, int, int, int, int, double, char[], char, char, char, double *);
static void trainLinearKernel(double *, double *, int, int, int, char, double *);
static void trainPolynomialKernel(double *, double *, int, int, int, int, char, char, double *);
static void trainLogisticKernel(double *, double *, int, int, int, char, double *);
static void trainGaussianKernel(double *, double *, int, int, int, double, char, char, double *);
void predictKernelMachineClassification(double *, int, char[], char, char, double *, int, int, int, double *);

#endif

