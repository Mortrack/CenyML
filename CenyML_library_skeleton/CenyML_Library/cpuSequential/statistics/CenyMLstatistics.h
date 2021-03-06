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
#ifndef CENYMLSTATISTICS_H
#define CENYMLSTATISTICS_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void getMean(double *, int, int, double *);
void getSort(char [], int, int, double *);
static void applyQuickSort(int, int, double *);
void getMedian(char [], double *, int, int, double *);
void getVariance(double *, int, int, int, double *);
void getStandardDeviation(double *, int, int, int, double *);
void getQuickMode(char [], int, int, double *, int *, double *);
void getMeanIntervals(double *, double *, float, char, int, int, double *);

#endif

