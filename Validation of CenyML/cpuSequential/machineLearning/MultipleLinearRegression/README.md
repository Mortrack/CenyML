
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation and evaluation of the multiple linear regression that was developed in the CenyML library. The following describes the purpose of each program contained in this directory:

- **main.c** creates a multiple linear regression model with the CenyML library to solve a multiple linear system.
- **scikitLearn.py** creates a multiple linear regression model with the scikit-learn library to solve a multiple linear system.

By comparing these algorithms, an additional goal is to make comparisons to get an idea of which algorithm obtains the best model and, in particular, which is faster and under what circumstances, if possible.

## How to compile and execute these files
For the following, it should be noted that after excecuting all these programs, either by console or outputted .png/.csv files, it is expected to have obtained some evaluation metrics and excecution times of all the processes performed in order to evaluate and compare them.

### How to compile the CenyML program:
1. Open the terminal and change its current working directory to the one that contains this README.md file (considering that the CenyML project files have not been changed).
2. The ".c" program that is contained in the current directory was made with the CenyML library. To compile it, enter the following commands into the terminal:
```console
$ make
```
NOTE: After these commands, an excecutable file with the name: "main.x" will be created, which has the same name as its ".c" file but that has the extension ".x" instead.
3. Run the compiled desired program made with the CenyML library with the command in the terminal window:
```console
$ ./main.x
```

### How to compile the scikit-learn program:
1. Compile and excecute the desired file, that was made with the scitkit-learn library, with your preferred method (the way I compiled and ran it was through the Spyder IDE).

## How to interpret the results obtained
Read the messages that were displayed in the terminal where you executed the chosen programs of this folder and analyze the results obtained. Review the documentation of the databases used, which is located in several .pdf files in the directory address "../../../../databases" with respect to the directory address of this README.md file. Compare the obtained models and determine if they correspond to the mathematical equation with which the databases were generated. In addition, compare all the evaluation metrics made and the processing times obtained with each tested program in order to obtain your own conclusions.

# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
