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


#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct singleNeuronDnnStruct_singleGPU {
	int gpuDevice;
	int maxUnrollingLoop;
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
extern "C" __global__ static void getTransposeOfInputData_singleGPU(double *, int, int, double *);
extern "C" __global__ static void getFxTilde_Au_dAu_and_accuracyTermsPart1_singleGPU(double *, double *, double *, int, int, int , double *, double *, double *, double *, double *);
extern "C" __device__ static void getFxTilde_singleGPU(double *, double *, int, double *, int, int);
extern "C" __global__ static void getErrorAndUpdateWeightValues_singleGPUpart1(double *, double *, int, int, double *, double *, double *);
extern "C"  __global__ static void getErrorAndUpdateWeightValues_singleGPUpart2(double *, int, int, int, int, int, double *);
extern "C" __device__ static void getActivationFunction_singleGPU(int, double *, double *, int);
extern "C" __device__ static void getDerivateOfActivationFunction_singleGPU(int, double *, double *, double *, int);
extern "C" __device__ static void getNeuronAdjustedCoefficientOfDetermination_singleGPUPart1(double *, double *, double *, double *, int);
extern "C" __global__ static void getParallelReduction_singleGPU(double *, double *, int, int, int);
extern "C" __device__ static void getDeviceParallelReduction_singleGPU(double *, double *, int, int, int);
extern "C" __global__ static void getNeuronAdjustedCoefficientOfDetermination_singleGPUvoidPart2(double *, int, double *, double *);
extern "C" void predictSingleNeuronDNN_singleGPU(struct singleNeuronDnnStruct_singleGPU *, double *);
extern "C" __global__ static void getPredictSingleNeuronDNN_singleGPU(double *, double *, int, int, int, int, double, int, int, double *, double *);

#else

#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct singleNeuronDnnStruct_singleGPU {
	int gpuDevice;
	int maxUnrollingLoop;
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
__global__ static void getTransposeOfInputData_singleGPU(double *, int, int, double *);
__global__ static void getFxTilde_Au_dAu_and_accuracyTermsPart1_singleGPU(double *, double *, double *, int, int, int , double *, double *, double *, double *, double *);
__device__ static void getFxTilde_singleGPU(double *, double *, int, double *, int, int);
__global__ static void getErrorAndUpdateWeightValues_singleGPUpart1(double *, double *, int, int, double *, double *, double *);
__global__ static void getErrorAndUpdateWeightValues_singleGPUpart2(double *, int, int, int, int, int, double *);
__device__ static void getActivationFunction_singleGPU(int, double *, double *, int);
__device__ static void getDerivateOfActivationFunction_singleGPU(int, double *, double *, double *, int);
__device__ static void getNeuronAdjustedCoefficientOfDetermination_singleGPUPart1(double *, double *, double *, double *, int);
__global__ static void getParallelReduction_singleGPU(double *, double *, int, int, int);
__device__ static void getDeviceParallelReduction_singleGPU(double *, double *, int, int, int);
__global__ static void getNeuronAdjustedCoefficientOfDetermination_singleGPUvoidPart2(double *, int, double *, double *);
void predictSingleNeuronDNN_singleGPU(struct singleNeuronDnnStruct_singleGPU *, double *);
__global__ static void getPredictSingleNeuronDNN_singleGPU(double *, double *, int, int, int, int, double, int, int, double *, double *);


#endif

