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

struct singleNeuronDnnStruct_multiGPU {
	int firstGpuDevice;
	int lastGpuDevice;
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

extern "C" void getSingleNeuronDNN_multiGPU(struct singleNeuronDnnStruct_multiGPU *);
extern "C" __global__ static void getTransposeOfInputData_multiGPU(double *, int, int, int, int, int, double *);
extern "C" __global__ static void getFxTilde_Au_dAu_and_accuracyTermsPart1_multiGPU(double *, double *, double *, int, int, int, int, double *, double *, double *, double *, double *);
extern "C" __device__ static void getFxTilde_multiGPU(double *, double *, int, int, double *, int, int);
extern "C" __global__ static void getErrorAndUpdateWeightValues_multiGPUpart1(double *, double *, int, int, double *, double *, double *);
extern "C"  __global__ static void getErrorAndUpdateWeightValues_multiGPUpart2(double *, int, int, int, int, int, double *);
extern "C" __device__ static void getActivationFunction_multiGPU(int, double *, double *, int);
extern "C" __device__ static void getDerivateOfActivationFunction_multiGPU(int, double *, double *, double *, int);
extern "C" __device__ static void getNeuronAdjustedCoefficientOfDetermination_multiGPUPart1(double *, double *, double *, double *, int);
extern "C" __global__ static void getParallelReduction_multiGPU(double *, double *, int, int, int);
extern "C" __device__ static void getDeviceParallelReduction_multiGPU(double *, double *, int, int, int);
extern "C" __global__ static void getNeuronAdjustedCoefficientOfDetermination_multiGPUvoidPart2(double *, int, double *, double *);
extern "C" void predictSingleNeuronDNN_multiGPU(struct singleNeuronDnnStruct_multiGPU *, double *);
extern "C" __global__ static void getPredictSingleNeuronDNN_multiGPU(double *, double *, int, int, int, int, int, double, int, int, double *, double *);

#else

#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

struct singleNeuronDnnStruct_multiGPU {
	int firstGpuDevice;
	int lastGpuDevice;
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

void getSingleNeuronDNN_multiGPU(struct singleNeuronDnnStruct_multiGPU *);
__global__ static void getTransposeOfInputData_multiGPU(double *, int, int, int, int, int, double *);
__global__ static void getFxTilde_Au_dAu_and_accuracyTermsPart1_multiGPU(double *, double *, double *, int, int, int, int, double *, double *, double *, double *, double *);
__device__ static void getFxTilde_multiGPU(double *, double *, int, int, double *, int, int);
__global__ static void getErrorAndUpdateWeightValues_multiGPUpart1(double *, double *, int, int, double *, double *, double *);
__global__ static void getErrorAndUpdateWeightValues_multiGPUpart2(double *, int, int, int, int, int, double *);
__device__ static void getActivationFunction_multiGPU(int, double *, double *, int);
__device__ static void getDerivateOfActivationFunction_multiGPU(int, double *, double *, double *, int);
__device__ static void getNeuronAdjustedCoefficientOfDetermination_multiGPUPart1(double *, double *, double *, double *, int);
__global__ static void getParallelReduction_multiGPU(double *, double *, int, int, int);
__device__ static void getDeviceParallelReduction_multiGPU(double *, double *, int, int, int);
__global__ static void getNeuronAdjustedCoefficientOfDetermination_multiGPUvoidPart2(double *, int, double *, double *);
void predictSingleNeuronDNN_multiGPU(struct singleNeuronDnnStruct_multiGPU *, double *);
__global__ static void getPredictSingleNeuronDNN_multiGPU(double *, double *, int, int, int, int, int, double, int, int, double *, double *);


#endif

