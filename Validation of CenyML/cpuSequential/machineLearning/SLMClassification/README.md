
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the method of interest that was developed in the CenyML library. The file "main.c" located in this directory contains the program that created a simple linear machine classification model with the CenyML library. The file "scikitLearn.py" contains a program that applies the Support Vector Machine (SVM) linear classifier to solve the same problem. The file "Dlib.py" contains the program to apply the SVM linear classifier also, but with the Python API provided by the Dlib library. Finally, the file "main.cpp" located in the directory "/DlibCplus/mainCode" contains the program for the linear SVM with the C++ version of the Dlib library.

It is important to note that no library was found that contained the simple linear machine classification algorithm as done in the CenyML library. However, because it was partly inspired by the SVM classifier, I decided to compare their performance. Also, it should be noted that after excecuting all these programs, either by console or outputted .png/.csv files, it is expected to have obtained some evaluation metrics and excecution times of all the processes performed in order to compare them. In addition, this data should also help to determine if the CenyML results were consistent or not.

## How to compile and execute these files
The instructions to excecute all programs mentioned are the following:
### How to compile the CenyML program:
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
### How to compile the scikit-learn program:
4. Compile and excecute the "scikitLearn.py" file with your preferred method (the way I compiled and ran it was through the Spyder IDE).
5. Read the messages that will be displayed in the terminal where you executed the "scikitLearn.py" file and analyze the plot made to determine if the CenyML results were consistent or not.
### How to compile the Dlib program made with its Python API:
6. Open the "Dlib.py" file with your preferred method (the way I compiled and ran it was through the Spyder IDE) and before compiling/excecuting it, make sure to manually enter, in the console, the code "import dlib". This is very important because due to the long directory address of the "Dlib.py" file, you will not be able to import the library as specified in the program (I will try to fix this in the future).
7. Compile and excecute the "Dlib.py" file.
8. Read the messages that will be displayed in the terminal where you executed the "Dlib.py" file and analyze them to determine if the CenyML results were consistent or not.
### How to compile the Dlib program made with C++:
9. Enter the directory address: "/DlibCplusplus/mainCode".
10. Enter the following commands into the terminal to compile the "main.cpp" file, containing the program made with the Dlib library in C++:
```console
$ make
```
NOTE: After these commands, an excecutable file named "main.x" will be created and will contain the program resulting from compiling the "main.cpp" file.
11. Enter the following commands into the terminal to excecute the compiled file:
```console
$ ./main.x
```
NOTE: After these commands, the program of the "main.cpp" file will be executed and some messages will be displayed in the terminal window regarding its obtained results and the execution times of each process. In addition, the results will be stored in one or more .csv and .png files.
12. Finally, compare all the results that were obtained and the excecution times between all the excecuted programs.
  
# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
