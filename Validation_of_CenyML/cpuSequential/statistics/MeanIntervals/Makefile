
# Directory path from the root of the project to the additionally added library files to complement the functionalities of the CenyML library.
oLib = CenyML_library_skeleton/otherLibraries
# Directory path from the root of the project to the CenyML library files location.
cM = CenyML_library_skeleton/CenyML_Library

# Set project directories for the Makefile to consider them.
Time_c = $(addprefix ../../../../,$(oLib)/time/mTime.c) # header file to measure time in Linux Ubuntu
Time_h = $(addprefix ../../../../,$(oLib)/time/mTime.h) # header file to measure time in Linux Ubuntu
# Time_c = $(addprefix ../../../../,$(oLib)/time/mTimeTer.c)
# Time_h = $(addprefix ../../../../,$(oLib)/time/mTimeTer.h)
csvManager_c = $(addprefix ../../../../,$(oLib)/csv/csvManager.c) # header file to measure time in Cygwin terminal window
csvManager_h = $(addprefix ../../../../,$(oLib)/csv/csvManager.h) # header file to measure time in Cygwin terminal window
CenyMLstatistics_c = $(addprefix ../../../../,$(cM)/cpuSequential/statistics/CenyMLstatistics.c)
CenyMLstatistics_h = $(addprefix ../../../../,$(cM)/cpuSequential/statistics/CenyMLstatistics.h)

# The following variables are to tell the compiler the desired flags for the compilation process.
OPTIMIZATIONLEVEL = -O3
SOFTWAREFLAGS = -lm
DEPENDENCYFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -c
TARGETFLAGS = $(OPTIMIZATIONLEVEL) $(SOFTWAREFLAGS) -o

# Monitor and compile all the files that are required for the target program to function.
# ----- THIS IS THE COMMAND TO COMPILE ALL THE MAIN FILES ----- #
main: main_95perCentMeanIntervals.o main_99perCentMeanIntervals.o main_99_9perCentMeanIntervals.o mTime.o csvManager.o CenyMLstatistics.o
	gcc main_95perCentMeanIntervals.o mTime.o csvManager.o CenyMLstatistics.o $(TARGETFLAGS) main_95perCentMeanIntervals.x
	gcc main_99perCentMeanIntervals.o mTime.o csvManager.o CenyMLstatistics.o $(TARGETFLAGS) main_99perCentMeanIntervals.x
	gcc main_99_9perCentMeanIntervals.o mTime.o csvManager.o CenyMLstatistics.o $(TARGETFLAGS) main_99_9perCentMeanIntervals.x

# ----- THESE ARE THE COMMANDS THAT MONITOR ANY CHANGES TO ANY MAIN FILE ----- #
main_95perCentMeanIntervals.o: main_95perCentMeanIntervals.c
	gcc $(DEPENDENCYFLAGS) main_95perCentMeanIntervals.c

main_99perCentMeanIntervals.o: main_99perCentMeanIntervals.c
	gcc $(DEPENDENCYFLAGS) main_99perCentMeanIntervals.c

main_99_9perCentMeanIntervals.o: main_99_9perCentMeanIntervals.c
	gcc $(DEPENDENCYFLAGS) main_99_9perCentMeanIntervals.c

# ----- THESE ARE THE COMMANDS THAT MONITOR ANY CHANGES TO ANY INCLUDED FILE ----- #
mTime.o: $(Time_c) $(Time_h)
	gcc $(DEPENDENCYFLAGS) $(Time_c)

csvManager.o: $(csvManager_c) $(csvManager_h)
	gcc $(DEPENDENCYFLAGS) $(csvManager_c)

CenyMLstatistics.o: $(CenyMLstatistics_c) $(CenyMLstatistics_h)
	gcc $(DEPENDENCYFLAGS) $(CenyMLstatistics_c)

# ----- THIS COMMAND IS USED TO CLEAN/REMOVE ALL THE COMPILATION GENERATED FILES ----- #
clean:
	rm *.o *.x

