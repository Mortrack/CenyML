# Create the following directory variables.
oLib = ../../../../"CenyML library skeleton"/otherLibraries
cM = ../../../../"CenyML library skeleton"/CenyML_Library

main.x:
	gcc main_linearKernel.c $(oLib)/time/mTimeTer.c $(oLib)/csv/csvManager.c $(oLib)/pbPlots/pbPlots.c $(oLib)/pbPlots/supportLib.c $(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c $(cM)/cpuSequential/machineLearning/CenyMLclassification.c -O3 -o main_linearKernel.x -lm
	gcc main_polynomialKernel.c $(oLib)/time/mTimeTer.c $(oLib)/csv/csvManager.c $(oLib)/pbPlots/pbPlots.c $(oLib)/pbPlots/supportLib.c $(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c $(cM)/cpuSequential/machineLearning/CenyMLclassification.c -O3 -o main_polynomialKernel.x -lm
	gcc main_logisticKernel.c $(oLib)/time/mTimeTer.c $(oLib)/csv/csvManager.c $(oLib)/pbPlots/pbPlots.c $(oLib)/pbPlots/supportLib.c $(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c $(cM)/cpuSequential/machineLearning/CenyMLclassification.c -O3 -o main_logisticKernel.x -lm
	gcc main_gaussianKernel.c $(oLib)/time/mTimeTer.c $(oLib)/csv/csvManager.c $(oLib)/pbPlots/pbPlots.c $(oLib)/pbPlots/supportLib.c $(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c $(cM)/cpuSequential/machineLearning/CenyMLclassification.c -O3 -o main_gaussianKernel.x -lm
