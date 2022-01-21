# Directory path from the root of the project to the additionally added library files to complement the functionalities of the CenyML library.
oLib = CenyML_library_skeleton/otherLibraries
# Directory path from the root of the project to the CenyML library files location.
cM = CenyML_library_skeleton/CenyML_Library
# Optimization compilation code request for the Tesla K80 GPU.
teslaK80_arch = -gencode arch=compute_37,code=sm_37
# Optimization compilation code request for the GeForce GTX 1660 SUPER GPU.
geForceGtx1660Super_arch = -gencode arch=compute_75,code=sm_75


# Set project directories for the Makefile to consider them.
Time_c = $(addprefix ../../../../,$(oLib)/time/mTime.c)
csvManager_c = $(addprefix ../../../../,$(oLib)/csv/csvManager.c)
pbPlots_c = $(addprefix ../../../../,$(oLib)/pbPlots/pbPlots.c)
supportLib_c = $(addprefix ../../../../,$(oLib)/pbPlots/supportLib.c)
CUDA_check_c = $(addprefix ../../../../,$(oLib)/cuda/CUDA_check.cu)
CenyMLregressionEvalMet_c = $(addprefix ../../../../,$(cM)/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.c)
CenyMLdeepLearning_SG = $(addprefix ../../../../,$(cM)/singleGpu/machineLearning/CenyMLdeepLearning_SG.cu)

# The following variables are to tell the compiler the desired flags for the compilation process.
OPTIMIZATIONLEVEL = -O3
SOFTWAREFLAGS = -lm
DEPENDENCYFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -c
TARGETFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -o

# Monitor and compile all the files that are required for the target program to function.
main_raisetoTheFirstPowerRegression.x: main_raisetoTheFirstPowerRegression.o Time.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o
	nvcc $(teslaK80_arch) main_raisetoTheFirstPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CUDA_check.o CenyMLregressionEvalMet.o CenyMLdeepLearning_SG.o $(TARGETFLAGS) main_raisetoTheFirstPowerRegression.x

main_raisetoTheFirstPowerRegression.o: main_raisetoTheFirstPowerRegression.c
	nvcc $(teslaK80_arch) $(DEPENDENCYFLAGS) main_raisetoTheFirstPowerRegression.c
	
Time.o: $(Time_c)
	nvcc $(teslaK80_arch) $(DEPENDENCYFLAGS) $(Time_c)

csvManager.o: $(csvManager_c)
	nvcc $(teslaK80_arch) $(DEPENDENCYFLAGS) $(csvManager_c)

pbPlots.o: $(pbPlots_c)
	nvcc $(teslaK80_arch) $(DEPENDENCYFLAGS) $(pbPlots_c)

supportLib.o: $(supportLib_c)
	nvcc $(teslaK80_arch) $(DEPENDENCYFLAGS) $(supportLib_c)

CUDA_check.o: $(CUDA_check_c)
	nvcc $(teslaK80_arch) $(DEPENDENCYFLAGS) $(CUDA_check_c)

CenyMLregressionEvalMet.o: $(CenyMLregressionEvalMet_c)
	nvcc $(teslaK80_arch) $(DEPENDENCYFLAGS) $(CenyMLregressionEvalMet_c)

CenyMLdeepLearning_SG.o: $(CenyMLdeepLearning_SG)
	nvcc $(teslaK80_arch) $(DEPENDENCYFLAGS) $(CenyMLdeepLearning_SG)

clean:
	rm *.o *.x
