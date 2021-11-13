
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the methods of interest that was developed in the CenyML library. Both the "main.c" and "main.py" files contain a program that applies such methods, where the former applies it through the CenyML library and the latter with a Python library. At the end, the console is expected to issue a message saying that the results match in case CenyML was able to get the correct results and if not, the console will say the opposite. At the same time, both programs will display in the console the execution times of all the processes performed so that they can also be compared and performance can be measured.

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
NOTE: The results will be stored in one or more .csv files and these should not be deleted, as they will be used in later steps in order to validate the CenyML function that was applied.
4. The program executed in step 3 will display some messages in the terminal window regarding its obtained results and the execution times of each process. In addition, it will also show if the reverse method currently being addressed matched with the comparative values or not.
5. Compile and excecute the "main.py" file with your preferred method (the original means for this step was to compile and run it through the Spyder IDE).
6. Read the messages that will be displayed in the terminal where you executed the "main.py" file and there it should be know if the feature scaling method applied with CenyML resulted in a match or not with the ones being compared to in Python.
  
# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
