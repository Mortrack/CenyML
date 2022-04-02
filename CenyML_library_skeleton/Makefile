
# Directory path to the additionally added library files to complement the functionalities of the CenyML library.
oLib = otherLibraries
# Directory path to the CenyML library files location.
cM = CenyML_Library
# Optimization compilation code request for the Tesla K80 GPU.
teslaK80_Optimization = -gencode arch=compute_37,code=sm_37
# Optimization compilation code request for the GeForce GTX 1660 SUPER GPU.
GeForceGTX1660SUPER_Optimization = -gencode arch=compute_75,code=sm_75
# Variable that defines the desired GPU optimization compilation.
gpuOptimization = $(teslaK80_Optimization)

# Set project directories for the Makefile to consider them.
Time_c = $(oLib)/time/mTime.c
Time_h = $(oLib)/time/mTime.h
csvManager_c = $(oLib)/csv/csvManager.c
csvManager_h = $(oLib)/csv/csvManager.h
pbPlots_c = $(oLib)/pbPlots/pbPlots.c
pbPlots_h = $(oLib)/pbPlots/pbPlots.h
supportLib_c = $(oLib)/pbPlots/supportLib.c
supportLib_h = $(oLib)/pbPlots/supportLib.h
CenyMLstatistics_c = $(cM)/cpuSequential/statistics/CenyMLstatistics.c
CenyMLstatistics_h = $(cM)/cpuSequential/statistics/CenyMLstatistics.h
CenyMLfeatureScaling_c = $(cM)/cpuSequential/featureScaling/CenyMLfeatureScaling.c
CenyMLfeatureScaling_h = $(cM)/cpuSequential/featureScaling/CenyMLfeatureScaling.h
CenyMLregressionEvalMet_c = $(cM)/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.c
CenyMLregressionEvalMet_h = $(cM)/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.h
CenyMLclassificationEvalMet_c = $(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c
CenyMLclassificationEvalMet_h = $(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.h
CenyMLregression_c = $(cM)/cpuSequential/machineLearning/CenyMLregression.c
CenyMLregression_h = $(cM)/cpuSequential/machineLearning/CenyMLregression.h
CenyMLclassification_c = $(cM)/cpuSequential/machineLearning/CenyMLclassification.c
CenyMLclassification_h = $(cM)/cpuSequential/machineLearning/CenyMLclassification.h
CenyMLdeepLearning_c = $(cM)/cpuSequential/machineLearning/CenyMLdeepLearning.c
CenyMLdeepLearning_h = $(cM)/cpuSequential/machineLearning/CenyMLdeepLearning.h
CenyMLdeepLearning_PC_c = $(cM)/cpuParallel/machineLearning/CenyMLdeepLearning_PC.c
CenyMLdeepLearning_PC_h = $(cM)/cpuParallel/machineLearning/CenyMLdeepLearning_PC.h
CenyMLdeepLearning_SG_cu = $(cM)/singleGpu/machineLearning/CenyMLdeepLearning_SG.cu
CenyMLdeepLearning_SG_h = $(cM)/singleGpu/machineLearning/CenyMLdeepLearning_SG.h
CenyMLdeepLearning_MG_cu = $(cM)/multiGpu/machineLearning/CenyMLdeepLearning_MG.cu
CenyMLdeepLearning_MG_h = $(cM)/multiGpu/machineLearning/CenyMLdeepLearning_MG.h

# The following variables are to tell the compiler the desired flags for the compilation process.
OPTIMIZATIONLEVEL = -O3
CUDA_PATH = /usr/local/cuda
SOFTWAREFLAGS = -I${CUDA_PATH}/samples/common/inc -I$(CUDA_PATH)/include -lpthread -lm
DEPENDENCYFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -dc
TARGETFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -o
NVCC = ${CUDA_PATH}/bin/nvcc

# Monitor and compile all the files that are required for the target program to function.
# ----- THIS IS THE COMMAND TO COMPILE ALL THE MAIN FILES ----- #
main.x: main.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLstatistics.o CenyMLfeatureScaling.o CenyMLregressionEvalMet.o CenyMLclassificationEvalMet.o CenyMLregression.o CenyMLclassification.o CenyMLdeepLearning.o CenyMLdeepLearning_PC.o CenyMLdeepLearning_SG.o CenyMLdeepLearning_MG.o
	$(NVCC) $(gpuOptimization) main.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLstatistics.o CenyMLfeatureScaling.o CenyMLregressionEvalMet.o CenyMLclassificationEvalMet.o CenyMLregression.o CenyMLclassification.o CenyMLdeepLearning.o CenyMLdeepLearning_PC.o CenyMLdeepLearning_SG.o CenyMLdeepLearning_MG.o $(TARGETFLAGS) main.x

# ----- THESE ARE THE COMMANDS THAT MONITOR ANY CHANGES TO ANY MAIN FILE ----- #
main.o: main.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main.c
	
# ----- THESE ARE THE COMMANDS THAT MONITOR ANY CHANGES TO ANY INCLUDED FILE ----- #
mTime.o: $(Time_c) $(Time_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(Time_c)

csvManager.o: $(csvManager_c) $(csvManager_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(csvManager_c)

pbPlots.o: $(pbPlots_c) $(pbPlots_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(pbPlots_c)

supportLib.o: $(supportLib_c) $(supportLib_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(supportLib_c)

CenyMLstatistics.o: $(CenyMLstatistics_c) $(CenyMLstatistics_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLstatistics_c)

CenyMLfeatureScaling.o: $(CenyMLfeatureScaling_c) $(CenyMLfeatureScaling_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLfeatureScaling_c)

CenyMLregressionEvalMet.o: $(CenyMLregressionEvalMet_c) $(CenyMLregressionEvalMet_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLregressionEvalMet_c)

CenyMLclassificationEvalMet.o: $(CenyMLclassificationEvalMet_c) $(CenyMLclassificationEvalMet_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLclassificationEvalMet_c)

CenyMLregression.o: $(CenyMLregression_c) $(CenyMLregression_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLregression_c)

CenyMLclassification.o: $(CenyMLclassification_c) $(CenyMLclassification_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLclassification_c)
	
CenyMLdeepLearning.o: $(CenyMLdeepLearning_c) $(CenyMLdeepLearning_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLdeepLearning_c)

CenyMLdeepLearning_PC.o: $(CenyMLdeepLearning_PC_c) $(CenyMLdeepLearning_PC_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLdeepLearning_PC_c)

CenyMLdeepLearning_SG.o: $(CenyMLdeepLearning_SG_cu) $(CenyMLdeepLearning_SG_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLdeepLearning_SG_cu)

CenyMLdeepLearning_MG.o: $(CenyMLdeepLearning_MG_cu) $(CenyMLdeepLearning_MG_h)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLdeepLearning_MG_cu)

# ----- THIS COMMAND IS USED TO CLEAN/REMOVE ALL THE COMPILATION GENERATED FILES ----- #
clean:
	rm *.o *.x

