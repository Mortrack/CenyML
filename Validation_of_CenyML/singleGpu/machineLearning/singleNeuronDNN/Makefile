# Directory path from the root of the project to the additionally added library files to complement the functionalities of the CenyML library.
oLib = CenyML_library_skeleton/otherLibraries
# Directory path from the root of the project to the CenyML library files location.
cM = CenyML_library_skeleton/CenyML_Library
# Optimization compilation code request for the Tesla K80 GPU.
teslaK80_Optimization = -gencode arch=compute_37,code=sm_37
# Optimization compilation code request for the GeForce GTX 1660 SUPER GPU.
GeForceGTX1660SUPER_Optimization = -gencode arch=compute_75,code=sm_75
# Variable that defines the desired GPU optimization compilation.
gpuOptimization = $(teslaK80_Optimization)

# Set project directories for the Makefile to consider them.
Time_c = $(addprefix ../../../../,$(oLib)/time/mTime.c)
csvManager_c = $(addprefix ../../../../,$(oLib)/csv/csvManager.c)
pbPlots_c = $(addprefix ../../../../,$(oLib)/pbPlots/pbPlots.c)
supportLib_c = $(addprefix ../../../../,$(oLib)/pbPlots/supportLib.c)
CUDA_check_c = $(addprefix ../../../../,$(oLib)/cuda/CUDA_check.cu)
CenyMLregressionEvalMet_c = $(addprefix ../../../../,$(cM)/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.c)
CenyMLclassificationEvalMet_c = $(addprefix ../../../../,$(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c)
CenyMLdeepLearning_SG = $(addprefix ../../../../,$(cM)/singleGpu/machineLearning/CenyMLdeepLearning_SG.cu)

# The following variables are to tell the compiler the desired flags for the compilation process.
OPTIMIZATIONLEVEL = -O3
CUDA_PATH = /usr/local/cuda
SOFTWAREFLAGS = -I${CUDA_PATH}/samples/common/inc -I$(CUDA_PATH)/include -lm
DEPENDENCYFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -dc
TARGETFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -o
NVCC = ${CUDA_PATH}/bin/nvcc

# Monitor and compile all the files that are required for the target program to function.
# ----- THIS IS THE COMMAND TO COMPILE ALL THE MAIN FILES ----- #
main: main_raisetoTheFirstPowerRegression.o main_firstOrderExponentialRegression.o main_hyperbolicTangentRegression.o main_linearClassification.o main_logisticRegression.o main_raiseToTheFifthPowerRegression.o main_raisetoTheFirstPowerRegression2.o main_raiseToTheFourthPowerRegression.o main_raiseToTheSecondPowerRegression.o main_raiseToTheSixthPowerRegression.o main_raiseToTheThirdPowerRegression.o main_secondOrderExponentialRegression.o main_simpleLinearRegressionWithRelu.o Time.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLclassificationEvalMet.o CenyMLdeepLearning_SG.o
	$(NVCC) $(gpuOptimization) main_raisetoTheFirstPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_raisetoTheFirstPowerRegression.x
	$(NVCC) $(gpuOptimization) main_firstOrderExponentialRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_firstOrderExponentialRegression.x
	$(NVCC) $(gpuOptimization) main_hyperbolicTangentRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_hyperbolicTangentRegression.x
	$(NVCC) $(gpuOptimization) main_linearClassification.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLclassificationEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_linearClassification.x
	$(NVCC) $(gpuOptimization) main_logisticRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_logisticRegression.x
	$(NVCC) $(gpuOptimization) main_raiseToTheFifthPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_raiseToTheFifthPowerRegression.x
	$(NVCC) $(gpuOptimization) main_raisetoTheFirstPowerRegression2.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_raisetoTheFirstPowerRegression2.x
	$(NVCC) $(gpuOptimization) main_raiseToTheFourthPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_raiseToTheFourthPowerRegression.x
	$(NVCC) $(gpuOptimization) main_raiseToTheSecondPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_raiseToTheSecondPowerRegression.x
	$(NVCC) $(gpuOptimization) main_raiseToTheSixthPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_raiseToTheSixthPowerRegression.x
	$(NVCC) $(gpuOptimization) main_raiseToTheThirdPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_raiseToTheThirdPowerRegression.x
	$(NVCC) $(gpuOptimization) main_secondOrderExponentialRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_secondOrderExponentialRegression.x
	$(NVCC) $(gpuOptimization) main_simpleLinearRegressionWithRelu.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_simpleLinearRegressionWithRelu.x

# ----- THESE ARE THE COMMANDS THAT MONITOR ANY CHANGES TO ANY MAIN FILE ----- #
main_raisetoTheFirstPowerRegression.o: main_raisetoTheFirstPowerRegression.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_raisetoTheFirstPowerRegression.c

main_firstOrderExponentialRegression.o: main_firstOrderExponentialRegression.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_firstOrderExponentialRegression.c

main_hyperbolicTangentRegression.o: main_hyperbolicTangentRegression.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_hyperbolicTangentRegression.c

main_linearClassification.o: main_linearClassification.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_linearClassification.c

main_logisticRegression.o: main_logisticRegression.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_logisticRegression.c

main_raiseToTheFifthPowerRegression.o: main_raiseToTheFifthPowerRegression.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_raiseToTheFifthPowerRegression.c

main_raisetoTheFirstPowerRegression2.o: main_raisetoTheFirstPowerRegression2.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_raisetoTheFirstPowerRegression2.c

main_raiseToTheFourthPowerRegression.o: main_raiseToTheFourthPowerRegression.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_raiseToTheFourthPowerRegression.c

main_raiseToTheSecondPowerRegression.o: main_raiseToTheSecondPowerRegression.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_raiseToTheSecondPowerRegression.c

main_raiseToTheSixthPowerRegression.o: main_raiseToTheSixthPowerRegression.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_raiseToTheSixthPowerRegression.c

main_raiseToTheThirdPowerRegression.o: main_raiseToTheThirdPowerRegression.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_raiseToTheThirdPowerRegression.c

main_secondOrderExponentialRegression.o: main_secondOrderExponentialRegression.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_secondOrderExponentialRegression.c

main_simpleLinearRegressionWithRelu.o: main_simpleLinearRegressionWithRelu.c
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) main_simpleLinearRegressionWithRelu.c
	
# ----- THESE ARE THE COMMANDS THAT MONITOR ANY CHANGES TO ANY INCLUDED FILE ----- #
Time.o: $(Time_c)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(Time_c)

csvManager.o: $(csvManager_c)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(csvManager_c)

pbPlots.o: $(pbPlots_c)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(pbPlots_c)

supportLib.o: $(supportLib_c)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(supportLib_c)

CUDA_check.o: $(CUDA_check_c)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CUDA_check_c)

CenyMLregressionEvalMet.o: $(CenyMLregressionEvalMet_c)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLregressionEvalMet_c)

CenyMLclassificationEvalMet.o: $(CenyMLclassificationEvalMet_c)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLclassificationEvalMet_c)
	
CenyMLdeepLearning_SG.o: $(CenyMLdeepLearning_SG)
	$(NVCC) $(gpuOptimization) $(DEPENDENCYFLAGS) $(CenyMLdeepLearning_SG)

# ----- THIS COMMAND IS USED TO CLEAN/REMOVE ALL THE COMPILATION GENERATED FILES ----- #
clean:
	rm *.o *.x
