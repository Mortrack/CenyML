# Create the following directory variables.
oLib = ../../../../"CenyML library skeleton"/otherLibraries
cM = ../../../../"CenyML library skeleton"/CenyML_Library

main_simpleLinearRegression.x:
	gcc main_simpleLinearRegression.c $(oLib)/time/mTimeTer.c $(oLib)/csv/csvManager.c $(oLib)/pbPlots/pbPlots.c $(oLib)/pbPlots/supportLib.c $(cM)/cpuSequential/evaluationMetrics/CenyMLregressionEvalMet.c $(cM)/cpuSequential/machineLearning/CenyMLdeepLearning.c -O3 -o main_simpleLinearRegression.x -lm
