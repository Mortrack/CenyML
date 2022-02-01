
# Directory path from the root of the project to the additionally added library files to complement the functionalities of the CenyML library.
oLib = CenyML_library_skeleton/otherLibraries
# Directory path from the root of the project to the CenyML library files location.
cM = CenyML_library_skeleton/CenyML_Library

# Set project directories for the Makefile to consider them.
Time_c = $(addprefix ../../../../,$(oLib)/time/mTime.c)
Time_h = $(addprefix ../../../../,$(oLib)/time/mTime.h)
csvManager_c = $(addprefix ../../../../,$(oLib)/csv/csvManager.c)
csvManager_h = $(addprefix ../../../../,$(oLib)/csv/csvManager.h)
pbPlots_c = $(addprefix ../../../../,$(oLib)/pbPlots/pbPlots.c)
pbPlots_h = $(addprefix ../../../../,$(oLib)/pbPlots/pbPlots.h)
supportLib_c = $(addprefix ../../../../,$(oLib)/pbPlots/supportLib.c)
supportLib_h = $(addprefix ../../../../,$(oLib)/pbPlots/supportLib.h)
CenyMLregressionEvalMet_c = $(addprefix ../../../../,$(cM)/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.c)
CenyMLregressionEvalMet_h = $(addprefix ../../../../,$(cM)/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.h)
CenyMLclassificationEvalMet_c = $(addprefix ../../../../,$(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c)
CenyMLclassificationEvalMet_h = $(addprefix ../../../../,$(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.h)
CenyMLdeepLearning_PC_c = $(addprefix ../../../../,$(cM)/cpuParallel/machineLearning/CenyMLdeepLearning_PC.c)
CenyMLdeepLearning_PC_h = $(addprefix ../../../../,$(cM)/cpuParallel/machineLearning/CenyMLdeepLearning_PC.h)

# The following variables are to tell the compiler the desired flags for the compilation process.
OPTIMIZATIONLEVEL = -O3
SOFTWAREFLAGS = -lpthread -lm
DEPENDENCYFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -c
TARGETFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -o

# Monitor and compile all the files that are required for the target program to function.
# ----- THIS IS THE COMMAND TO COMPILE ALL THE MAIN FILES ----- #
main: main_raisetoTheFirstPowerRegression.o main_firstOrderExponentialRegression.o main_hyperbolicTangentRegression.o main_linearClassification.o main_logisticRegression.o main_raiseToTheFifthPowerRegression.o main_raisetoTheFirstPowerRegression2.o main_raiseToTheFourthPowerRegression.o main_raiseToTheSecondPowerRegression.o main_raiseToTheSixthPowerRegression.o main_raiseToTheThirdPowerRegression.o main_secondOrderExponentialRegression.o main_simpleLinearRegressionWithRelu.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLclassificationEvalMet.o CenyMLdeepLearning_PC.o
	gcc main_raisetoTheFirstPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_raisetoTheFirstPowerRegression.x
	gcc main_firstOrderExponentialRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_firstOrderExponentialRegression.x
	gcc main_hyperbolicTangentRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_hyperbolicTangentRegression.x
	gcc main_linearClassification.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLclassificationEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_linearClassification.x
	gcc main_logisticRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_logisticRegression.x
	gcc main_raiseToTheFifthPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_raiseToTheFifthPowerRegression.x
	gcc main_raisetoTheFirstPowerRegression2.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_raisetoTheFirstPowerRegression2.x
	gcc main_raiseToTheFourthPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_raiseToTheFourthPowerRegression.x
	gcc main_raiseToTheSecondPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_raiseToTheSecondPowerRegression.x
	gcc main_raiseToTheSixthPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_raiseToTheSixthPowerRegression.x
	gcc main_raiseToTheThirdPowerRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_raiseToTheThirdPowerRegression.x
	gcc main_secondOrderExponentialRegression.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_secondOrderExponentialRegression.x
	gcc main_simpleLinearRegressionWithRelu.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLregressionEvalMet.o CenyMLdeepLearning_PC.o $(TARGETFLAGS) main_simpleLinearRegressionWithRelu.x

# ----- THESE ARE THE COMMANDS THAT MONITOR ANY CHANGES TO ANY MAIN FILE ----- #
main_raisetoTheFirstPowerRegression.o: main_raisetoTheFirstPowerRegression.c
	gcc $(DEPENDENCYFLAGS) main_raisetoTheFirstPowerRegression.c

main_firstOrderExponentialRegression.o: main_firstOrderExponentialRegression.c
	gcc $(DEPENDENCYFLAGS) main_firstOrderExponentialRegression.c

main_hyperbolicTangentRegression.o: main_hyperbolicTangentRegression.c
	gcc $(DEPENDENCYFLAGS) main_hyperbolicTangentRegression.c

main_linearClassification.o: main_linearClassification.c
	gcc $(DEPENDENCYFLAGS) main_linearClassification.c

main_logisticRegression.o: main_logisticRegression.c
	gcc $(DEPENDENCYFLAGS) main_logisticRegression.c

main_raiseToTheFifthPowerRegression.o: main_raiseToTheFifthPowerRegression.c
	gcc $(DEPENDENCYFLAGS) main_raiseToTheFifthPowerRegression.c

main_raisetoTheFirstPowerRegression2.o: main_raisetoTheFirstPowerRegression2.c
	gcc $(DEPENDENCYFLAGS) main_raisetoTheFirstPowerRegression2.c

main_raiseToTheFourthPowerRegression.o: main_raiseToTheFourthPowerRegression.c
	gcc $(DEPENDENCYFLAGS) main_raiseToTheFourthPowerRegression.c

main_raiseToTheSecondPowerRegression.o: main_raiseToTheSecondPowerRegression.c
	gcc $(DEPENDENCYFLAGS) main_raiseToTheSecondPowerRegression.c

main_raiseToTheSixthPowerRegression.o: main_raiseToTheSixthPowerRegression.c
	gcc $(DEPENDENCYFLAGS) main_raiseToTheSixthPowerRegression.c

main_raiseToTheThirdPowerRegression.o: main_raiseToTheThirdPowerRegression.c
	gcc $(DEPENDENCYFLAGS) main_raiseToTheThirdPowerRegression.c

main_secondOrderExponentialRegression.o: main_secondOrderExponentialRegression.c
	gcc $(DEPENDENCYFLAGS) main_secondOrderExponentialRegression.c

main_simpleLinearRegressionWithRelu.o: main_simpleLinearRegressionWithRelu.c
	gcc $(DEPENDENCYFLAGS) main_simpleLinearRegressionWithRelu.c
	
# ----- THESE ARE THE COMMANDS THAT MONITOR ANY CHANGES TO ANY INCLUDED FILE ----- #
mTime.o: $(Time_c) $(Time_h)
	gcc $(DEPENDENCYFLAGS) $(Time_c)

csvManager.o: $(csvManager_c) $(csvManager_h)
	gcc $(DEPENDENCYFLAGS) $(csvManager_c)

pbPlots.o: $(pbPlots_c) $(pbPlots_h)
	gcc $(DEPENDENCYFLAGS) $(pbPlots_c)

supportLib.o: $(supportLib_c) $(supportLib_h)
	gcc $(DEPENDENCYFLAGS) $(supportLib_c)

CenyMLregressionEvalMet.o: $(CenyMLregressionEvalMet_c) $(CenyMLregressionEvalMet_h)
	gcc $(DEPENDENCYFLAGS) $(CenyMLregressionEvalMet_c)

CenyMLclassificationEvalMet.o: $(CenyMLclassificationEvalMet_c) $(CenyMLclassificationEvalMet_h)
	gcc $(DEPENDENCYFLAGS) $(CenyMLclassificationEvalMet_c)
	
CenyMLdeepLearning_PC.o: $(CenyMLdeepLearning_PC_c) $(CenyMLdeepLearning_PC_h)
	gcc $(DEPENDENCYFLAGS) $(CenyMLdeepLearning_PC_c)

# ----- THIS COMMAND IS USED TO CLEAN/REMOVE ALL THE COMPILATION GENERATED FILES ----- #
clean:
	rm *.o *.x

