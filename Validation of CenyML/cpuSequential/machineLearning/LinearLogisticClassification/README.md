
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the method of interest that was developed in the CenyML library. Both the "main.c" and "scikitLearn.py" files contain a program that applies the same method, where the first one applies it through the CenyML library and the second one with a Python library. After running both programs, either by console or outputted .png/.csv files, it is expected to have obtained some evaluation metrics and execution times of all the processes performed so that they can also be compared. In addition, this data should also help to determine whether the CenyML results were consistent or not.

## How to compile and execute these files
The instructions to excecute both programs are the following:
1. Open the terminal and change its current working directory to this directory path.
2. Enter the following commands into the terminal to compile the "main.c" file:
```console
$ make
```
NOTE: After these commands, an excecutable file named "main.x" will be created and will contain the program resulting from compiling the "main.c" file.
3. Enter the following commands into the terminal to excecute the compiled file:
```console
$ ./main.x
```
NOTE: After these commands, the program of the "main.c" file will be executed and some messages will be displayed in the terminal window regarding its obtained results and the execution times of each process. In addition, the results will be stored in one or more .csv and .png files.
4. Compile and excecute the "scikitLearn.py" file with your preferred method (the original means for this step was to compile and run it through the Spyder IDE).
5. Read the messages that will be displayed in the terminal where you executed the "scikitLearn.py" file and analyze the plot made to determine if the CenyML results were consistent or not. As a complementary evaluation, compare the all the evaluation metrics done with both the CenyML library and scikit-learn.
  
# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
