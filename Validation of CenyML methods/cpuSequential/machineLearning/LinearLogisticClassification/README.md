
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the method of interest that was developed in the CenyML library. The "main.c" file contains a program that implements such method and, for this particular case, there are no other comparison programs/files, as no other library was found to contain this exact method. In this regard, scikit-learn provides the logistic classifier but calls it logistic regression for some reason. In addition, while making my best attempt to read and understand their documentation and implement their method as a regressor, I identified that it could only be possible with two machine learning features. Again, this is because, as stated in their documentation, their method was developed for classification purposes (not really for regression). Therefore, in order to make a fair evaluation, I decided not to use their method for this comparison and use it instead for the logistic classifier of the CenyML library. Nonetheless, after running the "main.c" program, either by console or generated .png/.csv files, it is expected to have obtained some evaluation metrics and execution times of all the processes performed. Consequently, this data should help to determine whether the CenyML results were consistent or not.

## How to compile and execute these files
The instructions to excecute the "main.c" program are the following:
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
4. Compare the reference equation with which the data was created with the one obtained with the CenyML library (the documentation is available within the folder of the databases) and together with all the data outputed by the program, determine if the results were consistent or not.
  
# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
