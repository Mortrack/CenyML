
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the method of interest that was developed in the CenyML library. The following describes the purpose of each program contained in this directory:
    - **main_linearKernel.c** creates a linear Kernel machine classification model with the CenyML library.
    - **main_polynomialKernel.c** creates a polynomial Kernel machine classification model with the CenyML library.
    - **main_logisticKernel.c** creates a logistic Kernel machine classification model with the CenyML library.
    - **main_gaussianKernel.c** creates a gaussian Kernel machine classification model with the CenyML library.
    - **polynomialKSVM_scikitLearn.py** creates a polynomial Kernel support vector machine model with the scikit-learn library.
    - **sigmoidKSVM_scikitLearn.py** creates a sigmoid Kernel support vector machine model with the scikit-learn library.
    - **radialBasisFunctionKSVM_scikitLearn.py** creates a radial basis function Kernel support vector machine model with the scikit-learn library.
    - **radialBasisFunctionKSVM_Dlib.py** creates a radial basis function Kernel support vector machine model with the Python API provided with the Dlib library.
    - **/DlibCplusplus/radialBasisFunctionKSVM/main.cpp** creates a radial basis function Kernel support vector machine model with the Dlib library in C++.

It is important to note that no library was found that contained the Kernel machine classification algorithm as done in the CenyML library. However, because it was partly inspired by the Kernel SVM classifier, I decided to compare their performance. Also, it should be noted that after excecuting all these programs, either by console or outputted .png/.csv files, it is expected to have obtained some evaluation metrics and excecution times of all the processes performed in order to compare them. In addition, this data should also help to determine if the CenyML results were consistent or not.

## How to compile and execute these files
The instructions to excecute all programs mentioned are the following:
### How to compile the CenyML programs:
1. Open the terminal and change its current working directory to the one that contains this README.md file (considering that the CenyML project files have not been changed).
2. Enter the following commands into the terminal to compile all the C files in which the CenyML programs were made:
```console
$ make
```
NOTE: After these commands, several excecutable files with the names: "main_linearKernel.x"; "main_polynomialKernel.x"; "main_logisticKernel.x" and; "main_gaussianKernel.x" will be created, where each of them will contain the compiled program of their respectives C files.
3. Enter the following commands into the terminal to excecute the compiled file "main_linearKernel.x":
```console
$ ./main_linearKernel.x
```
NOTE: After excecuting this and/or any other of the compiled C files, some messages will be displayed in the terminal window regarding their obtained results and the execution times of each process. In addition, the results will be stored in one or more .csv and .png files.
4. Enter the following commands into the terminal to excecute the compiled file "main_polynomialKernel.x":
```console
$ ./main_polynomialKernel.x
```
5. Enter the following commands into the terminal to excecute the compiled file "main_logisticKernel.x":
```console
$ ./main_logisticKernel.x
```
6. Enter the following commands into the terminal to excecute the compiled file "main_gaussianKernel.x":
```console
$ ./main_gaussianKernel.x
```

### How to compile the scikit-learn programs:
7. Compile and excecute the "polynomialKSVM_scikitLearn.py" file with your preferred method (the way I compiled and ran it was through the Spyder IDE).
8. Read the messages that will be displayed in the terminal where you executed the "polynomialKSVM_scikitLearn.py" file and analyze the results obtained to determine if the CenyML results were consistent or not.
9. Compile and excecute the "sigmoidKSVM_scikitLearn.py" file with your preferred method (the way I compiled and ran it was through the Spyder IDE).
10. Read the messages that will be displayed in the terminal where you executed the "sigmoidKSVM_scikitLearn.py" file and analyze the results obtained to determine if the CenyML results were consistent or not.
11. Compile and excecute the "radialBasisFunctionKSVM_scikitLearn.py" file with your preferred method (the way I compiled and ran it was through the Spyder IDE).
12. Read the messages that will be displayed in the terminal where you executed the "radialBasisFunctionKSVM_scikitLearn.py" file and analyze the results obtained to determine if the CenyML results were consistent or not.

### How to compile the Dlib program made with its Python API:
13. Open the "radialBasisFunctionKSVM_Dlib.py" file with your preferred method (the way I compiled and ran it was through the Spyder IDE).
14. Compile and excecute the "radialBasisFunctionKSVM_Dlib.py" file according to the means chosen in the previous step.
15. Read the messages that will be displayed in the terminal where you executed the "radialBasisFunctionKSVM_Dlib.py" file and analyze them to determine if the CenyML results were consistent or not.

### How to compile the Dlib program made with C++:
16. Enter the directory address through the terminal window: "/DlibCplusplus/radialBasisFunctionKSVM".
17. Enter the following commands into the terminal to compile the "main.cpp" file, containing the program made with the Dlib library in C++:
```console
$ make
```
NOTE: After these commands, an excecutable file named "main.x" will be created and will contain the program resulting from compiling the "main.cpp" file.
18. Enter the following commands into the terminal to excecute the compiled file:
```console
$ ./main.x
```
NOTE: After these commands, the program of the "main.cpp" file will be executed and some messages will be displayed in the terminal window regarding its obtained results and the execution times of each process. In addition, the results will be stored in one or more .csv and .png files.
19. Finally, compare all the results that were obtained and the excecution times between all the excecuted programs.
  
# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
