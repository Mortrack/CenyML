# Create the following directory variables.
oLib = ../../../../"CenyML library skeleton"/otherLibraries
cM = ../../../../"CenyML library skeleton"/CenyML_Library
time = mTime.c # header file to measure time in Linux Ubuntu
#time = mTimeTer.c # header file to measure time in Cygwin terminal window

main.x:
	gcc main_95perCentMeanIntervals.c $(oLib)/time/$(time) $(oLib)/csv/csvManager.c $(cM)/cpuSequential/statistics/CenyMLstatistics.c -O3 -o main_95perCentMeanIntervals.x -lm
	gcc main_99perCentMeanIntervals.c $(oLib)/time/$(time) $(oLib)/csv/csvManager.c $(cM)/cpuSequential/statistics/CenyMLstatistics.c -O3 -o main_99perCentMeanIntervals.x -lm
	gcc main_99_9perCentMeanIntervals.c $(oLib)/time/$(time) $(oLib)/csv/csvManager.c $(cM)/cpuSequential/statistics/CenyMLstatistics.c -O3 -o main_99_9perCentMeanIntervals.x -lm