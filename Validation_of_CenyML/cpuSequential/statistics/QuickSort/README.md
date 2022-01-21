
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the statistic method called quick sort and that was developed in the CenyML library. The following describes the purpose of each program contained in this directory:

- **main.c** applies the statistic method called quick sort, with the CenyML library and with respect to the random data of a gaussian system.
- **NumPy.py** applies the statistic method called quick sort, with the NumPy library and with respect to the random data of a gaussian system.

## How to compile and execute these files
For the following, it should be noted that after executing all these programs, either by console or outputted .csv files, it is expected to have obtained the corresponding results of the statistic methods that were applied and the execution times of all the processes performed.

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

### How to compile the NumPy program:
1. Compile and excecute the "NumPy.py" file, that was made with the NumPy library, with your preferred method (the way I compiled and ran it was through the Spyder IDE).

## How to interpret the results obtained
Read the messages that were displayed on the terminal in which you ran the programs chosen from this folder and analyze the results obtained. Compare the collected information and determine if they are consistent with the database that was used.

# Annexes

## WHY I DID NOT EVALUATE PERFORMANCE AND ONLY VALIDATED OBTAINING CORRECT RESULTS WITH THE "QUICK SORT" CENYML METHOD?
This method was only validated to give correct results but it was not validated in terms of performance. This is because the processing time of the quick sort method is highly dependent on the actual arrangement of the data in the input matrix where its best case time complexity is O(n*log(n)) and its worst one is O(n^2). Due to this fact, a fair performance evaluation between the CenyML method and the NumPy method would be to evaluate for every possible permutation case of the input matrix. Now, the thing is that this can be relatively quickly done in C under the CenyMl method, but on Python we would have to wait for a crazy amount of time (several hours or days) just to get one sample result due to the difference of processing time between C and Python when obtaining the permutations for each case of the input matrix. If there is a way to obtain comparable or faster processing times for the permutation process in Python, at least i was not able to obtain them neither writing RAW Python code or the NumPy library.

In conclusion, it was determined not to evaluate the performance on the quick sort method because i could not afford to spend waiting so much time to obtain each sample by permutation the input matrix in Python and because the quick sort method is not a machine learning method or strictly required in them, considering that the CenyML focuses on the machine learning methods primarly. However, the best attempt was made to obtain the fastest possible results with respect to the knowledge and experience that i have as a programmer.

# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
