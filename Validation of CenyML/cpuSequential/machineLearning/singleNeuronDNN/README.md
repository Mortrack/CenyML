
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the single neuron in Deep Neural Network that was developed in the CenyML library. The following describes the purpose of each program contained in this directory:

- **main_simpleLinearRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a simple linear regression problem.
- **main_simpleLinearRegression_tensorflow.py** creates a single single from an artifial neural network model from the class "Sequential()" with the tensorflow library to solve a simple linear regression problem.

It is important to note that the tensorflow model algorithm that was used, utilizes a different approach than the one used in the mathematical formulation done for the CenyML library. However, with respect to the time available and my understanding of the tensorflow documentation, I determined that the algorithm chosen was the most similar to the one used in the CenyML library. Therefore, when comparing these algorithms, it is unfair to automatically rule out that the faster one is the one that was programmed better due to their different solution approaches. However, these comparisons should give an idea of which algorithm obtains the best model and, in particular, which one is faster and under what circumstances, if possible.

## How to compile and execute these files
For the following, it should be noted that after excecuting all these programs, either by console or outputted .png/.csv files, it is expected to have obtained some evaluation metrics and excecution times of all the processes performed in order to evaluate and compare them.

### How to compile the CenyML programs:
1. Open the terminal and change its current working directory to the one that contains this README.md file (considering that the CenyML project files have not been changed).
2. All the .c programs that are contained in the current directory were all made with the CenyML library. To compile them, enter the following commands into the terminal:
```console
$ make
```
NOTE: After these commands, several excecutable files with the names: "main_simpleLinearRegression.x" will be created, where each of them will contain the compiled program of their respectives C files and with the same name but with the extension ".x". 
3. Run the compiled desired program made with the CenyML library with the command
```console
$ ./nameOfTheDesiredCenyMlProgram.x
``` 
NOTE: In place of "nameOfTheDesiredCenyMlProgram", place the actual name of the compiled program that you desire.
For example:
```console
$ ./main_simpleLinearRegression.x
```

### How to compile the tensorflow programs:
1. Compile and excecute the desired Python file, that was made with tensorflow, with your preferred method (the way I compiled and ran it was through the Spyder IDE).

## How to interpret the results obtained
Read the messages that were displayed in the terminal where you executed the chosen programs of this folder and analyze the results obtained. Review the documentation of the databases used, which is located in several .pdf files in the directory address "../../../../databases" with respect to the directory address of this README.md file. Compare the obtained models and determine if they correspond to the mathematical equation with which the databases were generated. In addition, compare all the evaluation metrics made and the processing times obtained with each tested program in order to obtain your own conclusions.

# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
