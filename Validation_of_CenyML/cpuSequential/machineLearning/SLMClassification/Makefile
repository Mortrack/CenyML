
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
CenyMLclassificationEvalMet_c = $(addprefix ../../../../,$(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c)
CenyMLclassificationEvalMet_h = $(addprefix ../../../../,$(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.h)
CenyMLclassification_c = $(addprefix ../../../../,$(cM)/cpuSequential/machineLearning/CenyMLclassification.c)
CenyMLclassification_h = $(addprefix ../../../../,$(cM)/cpuSequential/machineLearning/CenyMLclassification.h)

# The following variables are to tell the compiler the desired flags for the compilation process.
OPTIMIZATIONLEVEL = -O3
SOFTWAREFLAGS = -lm
DEPENDENCYFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -c
TARGETFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -o

# Monitor and compile all the files that are required for the target program to function.
# ----- THIS IS THE COMMAND TO COMPILE ALL THE MAIN FILES ----- #
main.x: main.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLclassificationEvalMet.o CenyMLclassification.o
	gcc main.o mTime.o csvManager.o pbPlots.o supportLib.o CenyMLclassificationEvalMet.o CenyMLclassification.o $(TARGETFLAGS) main.x
	
# ----- THESE ARE THE COMMANDS THAT MONITOR ANY CHANGES TO ANY MAIN FILE ----- #
main.o: main.c
	gcc $(DEPENDENCYFLAGS) main.c
	
# ----- THESE ARE THE COMMANDS THAT MONITOR ANY CHANGES TO ANY INCLUDED FILE ----- #
mTime.o: $(Time_c) $(Time_h)
	gcc $(DEPENDENCYFLAGS) $(Time_c)

csvManager.o: $(csvManager_c) $(csvManager_h)
	gcc $(DEPENDENCYFLAGS) $(csvManager_c)

pbPlots.o: $(pbPlots_c) $(pbPlots_h)
	gcc $(DEPENDENCYFLAGS) $(pbPlots_c)

supportLib.o: $(supportLib_c) $(supportLib_h)
	gcc $(DEPENDENCYFLAGS) $(supportLib_c)

CenyMLclassificationEvalMet.o: $(CenyMLclassificationEvalMet_c) $(CenyMLclassificationEvalMet_h)
	gcc $(DEPENDENCYFLAGS) $(CenyMLclassificationEvalMet_c)
	
CenyMLclassification.o: $(CenyMLclassification_c) $(CenyMLclassification_h)
	gcc $(DEPENDENCYFLAGS) $(CenyMLclassification_c)

# ----- THIS COMMAND IS USED TO CLEAN/REMOVE ALL THE COMPILATION GENERATED FILES ----- #
clean:
	rm *.o *.x

