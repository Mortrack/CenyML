# Create the following directory variables.
oLib = ../../../../"CenyML library skeleton"/otherLibraries
cM = ../../../../"CenyML library skeleton"/CenyML_Library
time = mTime.c # header file to measure time in Linux Ubuntu
#time = mTimeTer.c # header file to measure time in Cygwin terminal window

main.x:
	gcc main.c $(oLib)/time/$(time) $(oLib)/csv/csvManager.c $(oLib)/pbPlots/pbPlots.c $(oLib)/pbPlots/supportLib.c $(cM)/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.c $(cM)/cpuSequential/machineLearning/CenyMLregression.c -O3 -o main.x -lm