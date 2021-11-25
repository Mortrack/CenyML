
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the method of interest that was developed in the CenyML library. Both the "main.c" and "main.py" files contain a program that applies such method, where the former applies it through the CenyML library and the latter with a Python library. At the end, the console is expected to issue a message saying that the results match in case CenyML was able to get the correct results and if not, the console will say the opposite. At the same time, both programs will display in the console the execution times of all the processes performed so that they can also be compared and performance can be measured.

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
NOTE: After these commands, the program of the "main.c" file will be executed and some messages will be displayed in the terminal window regarding its obtained results and the execution times of each process. In addition, the results will be stored in one or more .csv files and these should not be deleted, as they will be used in later steps in order to validate the CenyML function that was applied.
4. Compile and excecute the "main.py" file with your preferred method (the original means for this step was to compile and run it through the Spyder IDE).
5. Read the messages that will be displayed in the terminal where you executed the "main.py" file and there it should be know if the CenyML results matched or not with the ones being compared to in Python.

# Annexes

## WHY I DID NOT EVALUATE PERFORMANCE AND ONLY VALIDATED OBTAINING CORRECT RESULTS WITH THE "QUICK SORT" CENYML METHOD?
This method was only validated to give correct results but it was not validated in terms of performance. This is because the processing time of the quick sort method is highly dependent on the actual arrangement of the data in the input matrix where its best case time complexity is O(n*log(n)) and its worst one is O(n^2). Due to this fact, a fair performance evaluation between the CenyML method and the NumPy method would be to evaluate for every possible permutation case of the input matrix. Now, the thing is that this can be relatively quickly done in C under the CenyMl method, but on Python we would have to wait for a crazy amount of time (several hours or days) just to get one sample result due to the difference of processing time between C and Python when obtaining the permutations for each case of the input matrix. If there is a way to obtain comparable or faster processing times for the permutation process in Python, at least i was not able to obtain them neither writing RAW Python code or the NumPy library.

In conclusion, it was determined not to evaluate the performance on the quick sort method because i could not afford to spend waiting so much time to obtain each sample by permutation the input matrix in Python and because the quick sort method is not a machine learning method or strictly required in them, considering that the CenyML focuses on the machine learning methods primarly. However, the best attempt was made to obtain the fastest possible results with respect to the knowledge and experience that i have as a programmer.

# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
