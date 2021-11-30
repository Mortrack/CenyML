
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The file located in this directory path is used for the validation and evaluation of the logistic regression that was developed in the CenyML library. The following describes the purpose of such program, which is contained in this directory:

- **main.c** creates a logistic regression model with the CenyML library to solve a logistic system.

It is important to note that it was desired to compare this algorithm with an existing alternative. However, with respect to the available time I had to do research, I was unable to find any machine learning library that had an algorithm such as the one mathematically formulated in the CenyML library. In this sense, the only means to generate a solution for the proposed problem to be solved was through the deep learning solutions provided by other libraries. However, they differed completely mathematically in the way they solve the problem and I considered it biased to compare them and publish such results, which is why i did not compared this CenyML library algorithm with others.

## How to compile and execute these files
For the following, it should be noted that after excecuting the .c program, either by console or outputted .png/.csv files, it is expected to have obtained some evaluation metrics and excecution times of all the processes performed in order to evaluate them.

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

## How to interpret the results obtained
Read the messages that were displayed in the terminal where you executed the program of this folder and analyze the results obtained. Review the documentation of the databases used, which is located in several .pdf files in the directory address "../../../../databases" with respect to the directory address of this README.md file. Compare the obtained model and determine if it corresponds to the mathematical equation with which the databases were generated. In addition, compare all the evaluation metrics made and the processing times obtained in order to obtain your own conclusions.

# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
