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
#ifndef CENYMLDEEPLEARNING_PC_H
#define CENYMLDEEPLEARNING_PC_H

#include <pthread.h> // POSIX thread library that allows you to program with CPU parallelism.
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct singleNeuronDnnStruct_parallelCPU {
   int cpuThreads;
   double *X;
   double *w_first;
   double *Y;
   int n;
   int m;
   int p;
   char isInitial_w;
   char isClassification;
   double threshold;
   int desiredValueForGroup1;
   int desiredValueForGroup2;
   int activationFunctionToBeUsed;
   double learningRate;
   double stopAboveThisAccuracy;
   int maxEpochs;
   char isReportLearningProgress;
   int reportEachSpecifiedEpochs;
   double *w_best;
   double bestAccuracy;
   double *w_new;
};

void getSingleNeuronDNN_parallelCPU(struct singleNeuronDnnStruct_parallelCPU *);
static void *getTransposeOfInputData(void *);
static void *getFxTilde_Au_dAu_and_accuracyTermsPart1(void *);
static void *getErrorAndUpdateWeightValues(void *);
static void getReluActivation_parallelCPU(void *);
static void getDerivateReluActivation_parallelCPU(void *);
static void getTanhActivation_parallelCPU(void *);
static void getDerivateTanhActivation_parallelCPU(void *);
static void getLogisticActivation_parallelCPU(void *);
static void getDerivateLogisticActivation_parallelCPU(void *);
static void getRaiseToTheFirstPowerActivation_parallelCPU(void *);
static void getDerivateRaiseToTheFirstPowerActivation_parallelCPU(void *);
static void getRaiseToTheSecondPowerActivation_parallelCPU(void *);
static void getDerivateRaiseToTheSecondPowerActivation_parallelCPU(void *);
static void getRaiseToTheThirdPowerActivation_parallelCPU(void *);
static void getDerivateRaiseToTheThirdPowerActivation_parallelCPU(void *);
static void getRaiseToTheFourthPowerActivation_parallelCPU(void *);
static void getDerivateRaiseToTheFourthPowerActivation_parallelCPU(void *);
static void getRaiseToTheFifthPowerActivation_parallelCPU(void *);
static void getDerivateRaiseToTheFifthPowerActivation_parallelCPU(void *);
static void getRaiseToTheSixthPowerActivation_parallelCPU(void *);
static void getDerivateRaiseToTheSixthPowerActivation_parallelCPU(void *);
static void getFirstOrderDegreeExponentialActivation_parallelCPU(void *);
static void getDerivateFirstOrderDegreeExponentialActivation_parallelCPU(void *);
static void getSecondOrderDegreeExponentialActivation_parallelCPU(void *);
static void getDerivateSecondOrderDegreeExponentialActivation_parallelCPU(void *);
static void getNeuronAdjustedCoefficientOfDetermination_parallelCPUvoidPart1(void *);
static void *getNeuronAdjustedCoefficientOfDetermination_parallelCPUvoidPart2(void *);
static void getNeuronAccuracy_parallelCPU(double *, double *, int, double *);

static void getNeuronAdjustedCoefficientOfDetermination(double *, double *, int, int, int, double *);

void predictSingleNeuronDNN_parallelCPU(struct singleNeuronDnnStruct_parallelCPU *, double *);

#endif

