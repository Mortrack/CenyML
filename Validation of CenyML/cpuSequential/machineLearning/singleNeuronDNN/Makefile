# Create the following directory variables.
oLib = ../../../../"CenyML library skeleton"/otherLibraries
cM = ../../../../"CenyML library skeleton"/CenyML_Library
time = mTime.c # header file to measure time in Linux Ubuntu
#time = mTimeTer.c # header file to measure time in Cygwin terminal window

main_simpleLinearRegression.x:
	gcc main_simpleLinearRegression.c $(oLib)/time/$(time) $(oLib)/csv/csvManager.c $(oLib)/pbPlots/pbPlots.c $(oLib)/pbPlots/supportLib.c $(cM)/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.c $(cM)/cpuSequential/machineLearning/CenyMLdeepLearning.c -O3 -o main_simpleLinearRegression.x -lm
