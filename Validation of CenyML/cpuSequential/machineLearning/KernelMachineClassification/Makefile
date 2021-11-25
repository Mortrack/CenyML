# Create the following directory variables.
oLib = ../../../../"CenyML library skeleton"/otherLibraries
cM = ../../../../"CenyML library skeleton"/CenyML_Library

main.x:
	gcc main.c $(oLib)/time/mTimeTer.c $(oLib)/csv/csvManager.c $(oLib)/pbPlots/pbPlots.c $(oLib)/pbPlots/supportLib.c $(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c $(cM)/cpuSequential/machineLearning/CenyMLclassification.c -O3 -o main.x -lm
