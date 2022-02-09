
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation and evaluation of the regression evaluation metric called mean squared error and that was developed in the CenyML library. The following describes the purpose of each program contained in this directory:

- **main.c** applies the regression evaluation metric called mean squared error, with the CenyML library and with respect to the real and random data of a linear regression system.
- **scikitLearn.py** applies the regression evaluation metric called mean squared error, with the scikit-learn library and with respect to the real and random data of a linear regression system.

By comparing these algorithms, an additional goal is to make comparisons to get an idea of which algorithm obtains faster results and under what circumstances, if possible.

## How to compile and execute these files
For the following, it should be noted that after executing all these programs, either by console or outputted .csv files, it is expected to have obtained the corresponding results of the evaluation metrics that were applied and the execution times of all the processes performed in order to evaluate and compare them.

### How to compile the CenyML program:
1. Open the terminal and change its current working directory to the one that contains this README.md file (considering that the CenyML project files have not been changed).
2. The ".c" program that is contained in the current directory was made with the CenyML library. To compile it, enter the following commands into the terminal:

```console
$ make
```

**NOTE:** After these commands, an executable file with the name: "main.x" will be created, which has the same name as its ".c" file but that has the extension ".x" instead.
3. Run the compiled desired program made with the CenyML library with the command in the terminal window:

```console
$ ./main.x
```

**NOTE:** Do not delete the generated .csv file because it will be used by the Python program for validation purposes.

### How to compile the scikit-learn program:
1. Compile and execute the "scikitLearn.py" file, that was made with the scitkit-learn library, with your preferred method (the way I compiled and ran it was through the Spyder IDE).

## How to interpret the results obtained
Read the messages that were displayed on the terminal in which you ran the programs chosen from this folder and analyze the results obtained. Compare the collected information and determine if they are consistent with the databases that were used. Also, compare the processing times obtained with each program tested to obtain your own conclusions.

# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
