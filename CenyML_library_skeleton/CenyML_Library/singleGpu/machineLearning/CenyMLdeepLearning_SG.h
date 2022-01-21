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
#ifdef __cplusplus


//#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct singleNeuronDnnStruct_singleGPU {
	int gpuDevice;
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

extern "C" void getSingleNeuronDNN_singleGPU(struct singleNeuronDnnStruct_singleGPU *);
extern "C" static void *getTransposeOfInputData_singleGPU(void *);
extern "C" static void *getFxTilde_Au_dAu_and_accuracyTermsPart1_singleGPU(void *);
extern "C" static void *getErrorAndUpdateWeightValues_singleGPU(void *);
extern "C" static void getReluActivation_singleGPU(void *);
extern "C" static void getDerivateReluActivation_singleGPU(void *);
extern "C" static void getTanhActivation_singleGPU(void *);
extern "C" static void getDerivateTanhActivation_singleGPU(void *);
extern "C" static void getLogisticActivation_singleGPU(void *);
extern "C" static void getDerivateLogisticActivation_singleGPU(void *);
extern "C" static void getRaiseToTheFirstPowerActivation_singleGPU(void *);
extern "C" static void getDerivateRaiseToTheFirstPowerActivation_singleGPU(void *);
extern "C" static void getRaiseToTheSecondPowerActivation_singleGPU(void *);
extern "C" static void getDerivateRaiseToTheSecondPowerActivation_singleGPU(void *);
extern "C" static void getRaiseToTheThirdPowerActivation_singleGPU(void *);
extern "C" static void getDerivateRaiseToTheThirdPowerActivation_singleGPU(void *);
extern "C" static void getRaiseToTheFourthPowerActivation_singleGPU(void *);
extern "C" static void getDerivateRaiseToTheFourthPowerActivation_singleGPU(void *);
extern "C" static void getRaiseToTheFifthPowerActivation_singleGPU(void *);
extern "C" static void getDerivateRaiseToTheFifthPowerActivation_singleGPU(void *);
extern "C" static void getRaiseToTheSixthPowerActivation_singleGPU(void *);
extern "C" static void getDerivateRaiseToTheSixthPowerActivation_singleGPU(void *);
extern "C" static void getFirstOrderDegreeExponentialActivation_singleGPU(void *);
extern "C" static void getDerivateFirstOrderDegreeExponentialActivation_singleGPU(void *);
extern "C" static void getSecondOrderDegreeExponentialActivation_singleGPU(void *);
extern "C" static void getDerivateSecondOrderDegreeExponentialActivation_singleGPU(void *);
extern "C" static void getNeuronAdjustedCoefficientOfDetermination_singleGPUvoidPart1(void *);
extern "C" static void *getNeuronAdjustedCoefficientOfDetermination_singleGPUvoidPart2(void *);
extern "C" void predictSingleNeuronDNN_singleGPU(struct singleNeuronDnnStruct_singleGPU *, double *);
extern "C" static void *getPredictSingleNeuronDNN_singleGPU(void *);


#else

//#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct singleNeuronDnnStruct_singleGPU {
	int gpuDevice;
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

void getSingleNeuronDNN_singleGPU(struct singleNeuronDnnStruct_singleGPU *);
static void *getTransposeOfInputData_singleGPU(void *);
static void *getFxTilde_Au_dAu_and_accuracyTermsPart1_singleGPU(void *);
static void *getErrorAndUpdateWeightValues_singleGPU(void *);
static void getReluActivation_singleGPU(void *);
static void getDerivateReluActivation_singleGPU(void *);
static void getTanhActivation_singleGPU(void *);
static void getDerivateTanhActivation_singleGPU(void *);
static void getLogisticActivation_singleGPU(void *);
static void getDerivateLogisticActivation_singleGPU(void *);
static void getRaiseToTheFirstPowerActivation_singleGPU(void *);
static void getDerivateRaiseToTheFirstPowerActivation_singleGPU(void *);
static void getRaiseToTheSecondPowerActivation_singleGPU(void *);
static void getDerivateRaiseToTheSecondPowerActivation_singleGPU(void *);
static void getRaiseToTheThirdPowerActivation_singleGPU(void *);
static void getDerivateRaiseToTheThirdPowerActivation_singleGPU(void *);
static void getRaiseToTheFourthPowerActivation_singleGPU(void *);
static void getDerivateRaiseToTheFourthPowerActivation_singleGPU(void *);
static void getRaiseToTheFifthPowerActivation_singleGPU(void *);
static void getDerivateRaiseToTheFifthPowerActivation_singleGPU(void *);
static void getRaiseToTheSixthPowerActivation_singleGPU(void *);
static void getDerivateRaiseToTheSixthPowerActivation_singleGPU(void *);
static void getFirstOrderDegreeExponentialActivation_singleGPU(void *);
static void getDerivateFirstOrderDegreeExponentialActivation_singleGPU(void *);
static void getSecondOrderDegreeExponentialActivation_singleGPU(void *);
static void getDerivateSecondOrderDegreeExponentialActivation_singleGPU(void *);
static void getNeuronAdjustedCoefficientOfDetermination_singleGPUvoidPart1(void *);
static void *getNeuronAdjustedCoefficientOfDetermination_singleGPUvoidPart2(void *);
void predictSingleNeuronDNN_singleGPU(struct singleNeuronDnnStruct_singleGPU *, double *);
static void *getPredictSingleNeuronDNN_singleGPU(void *);


#endif

