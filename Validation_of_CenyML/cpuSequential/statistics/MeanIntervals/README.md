
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation and evaluation of the statistic method called mean intervals and that was developed in the CenyML library. The following describes the purpose of each program contained in this directory:

- **main_95perCentMeanIntervals.c** applies the statistic method called mean intervals with a trust interval of 95% through the CenyML library and with respect to the random data of a Multiple polynomial system csv file.
- **main_99perCentMeanIntervals.c** applies the statistic method called mean intervals with a trust interval of 99% through the CenyML library and with respect to the random data of a Multiple polynomial system csv file.
- **main_99_9perCentMeanIntervals.c** applies the statistic method called mean intervals with a trust interval of 99.9% through the CenyML library and with respect to the random data of a Multiple polynomial system csv file.

When executing these programs, the data of some .csv files will be extracted for comparison purposes. The name of those files are: 1) 95perCentMeanIntervals_ExcelResults.csv; 2) 99perCentMeanIntervals_ExcelResults.csv and 3) 99_9perCentMeanIntervals_ExcelResults.csv. All of these files contain the results that were obtained when using the "Data Analysis Tool" of Excel for the same input data as the CenyML programs use. The goal of this is only to validate if the CenyML library is applying the mean intervals correctly and no performance evaluation will be made for this particular algorithm (there were no libraries found that could apply this algorithm as proposed in the CenyML library).

## How to compile and execute these files
For the following, it should be noted that after excecuting all these programs, either by console or outputted .csv files, it is expected to have obtained the corresponding results of the statistic methods that were applied and the excecution times of all the processes performed in order to evaluate and compare them.

### How to compile the CenyML program:
1. Open the terminal and change its current working directory to the one that contains this README.md file (considering that the CenyML project files have not been changed).
2. All the .c programs that are contained in the current directory were all made with the CenyML library. To compile them, enter the following commands into the terminal:
```console
$ make
```
**NOTE:** After these commands, several executable files with the names: "main_95perCentMeanIntervals.x"; "main_99perCentMeanIntervals.x" and; "main_99_9perCentMeanIntervals.x" will be created, where each of them will contain the compiled program of their respectives C files but with the extension ".x".

3. Run the compiled desired program made with the CenyML library with the following command in the terminal window:
```console
$ ./nameOfTheProgram.x
```

**NOTE:** Substitute "nameOfTheProgram" for the particular name of the compiled file you want to excecute.
For example:

```console
$ ./main_95perCentMeanIntervals.x
``` 

## How to interpret the results obtained
Read the messages that were displayed on the terminal in which you ran the programs chosen from this folder and analyze the results obtained. Compare the collected information and determine if they are consistent with the database that was used.

# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
