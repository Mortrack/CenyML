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
#ifndef CENYMLDEEPLEARNING_H
#define CENYMLDEEPLEARNING_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct singleNeuronDnnStruct {
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

void getSingleNeuronDNN(struct singleNeuronDnnStruct *);
static void getReluActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateReluActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getTanhActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateTanhActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getLogisticActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateLogisticActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getRaiseToTheFirstPowerActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateRaiseToTheFirstPowerActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getSquareRootActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateSquareRootActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getRaiseToTheSecondPowerActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateRaiseToTheSecondPowerActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getRaiseToTheThirdPowerActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateRaiseToTheThirdPowerActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getRaiseToTheFourthPowerActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateRaiseToTheFourthPowerActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getRaiseToTheFifthPowerActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateRaiseToTheFifthPowerActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getRaiseToTheSixthPowerActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateRaiseToTheSixthPowerActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getFirstOrderDegreeExponentialActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateFirstOrderDegreeExponentialActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getSecondOrderDegreeExponentialActivation(double *, double *, struct singleNeuronDnnStruct *);
static void getDerivateSecondOrderDegreeExponentialActivation(double *, double *, double *, struct singleNeuronDnnStruct *);
static void getNeuronAdjustedCoefficientOfDetermination(double *, double *, int, int, int, double *);
static void getNeuronAccuracy(double *, double *, int, double *);
void predictSingleNeuronDNN(struct singleNeuronDnnStruct *, double *);

#endif

