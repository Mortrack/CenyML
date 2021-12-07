# Create the following directory variables.
oLib = ../../../../../../"CenyML library skeleton"/otherLibraries
cM = ../../../../../../"CenyML library skeleton"/CenyML_Library
time = mTime.c # header file to measure time in Linux Ubuntu
#time = mTimeTer.c # header file to measure time in Cygwin terminal window

main.x:
	g++ -std=c++11 -O3 -I.. $(oLib)/time/$(time) $(oLib)/csv/csvManager.c $(oLib)/pbPlots/pbPlots.c $(oLib)/pbPlots/supportLib.c $(cM)/cpuSequential/evaluationMetrics/CenyMLclassificationEvalMet.c ../dlib/all/source.cpp -lpthread -lX11 main.cpp -o main.x -lm

