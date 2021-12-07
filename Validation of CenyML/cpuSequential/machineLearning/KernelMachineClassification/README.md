
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the Kernel machine classification that was developed in the CenyML library. The following describes the purpose of each program contained in this directory:

- **main_linearKernel.c** creates a linear Kernel machine classification model with the CenyML library to solve a linear classification problem.
- **main_polynomialKernel.c** creates a polynomial Kernel machine classification model with the CenyML library to solve a polynomial classification problem.
- **main_logisticKernel.c** creates a logistic Kernel machine classification model with the CenyML library to solve a linear classification problem.
- **main_gaussianKernel.c** creates a gaussian Kernel machine classification model with the CenyML library to solve a gaussian classification problem.
- **polynomialKSVM_scikitLearn.py** creates a polynomial Kernel support vector machine model with the scikit-learn library to solve a polynomial classification problem.
- **sigmoidKSVM_scikitLearn.py** creates a sigmoid Kernel support vector machine model with the scikit-learn library to solve a linear classification problem.
- **radialBasisFunctionKSVM_scikitLearn.py** creates a radial basis function Kernel support vector machine model with the scikit-learn library to solve a gaussian classification problem.
- **radialBasisFunctionKSVM_Dlib.py** creates a radial basis function Kernel support vector machine model with the Python API provided with the Dlib library to solve a gaussian classification problem.
- **/DlibCplusplus/radialBasisFunctionKSVM/main.cpp** creates a radial basis function Kernel support vector machine model with the Dlib library in C++ to solve a gaussian classification problem.

It is important to note that no library was found to have the Kernel machine classification algorithm as done in the CenyML library. However, because it was partly inspired by the Kernel support vector machine classifier, I decided to compare its performance with some representative libraries applying the Kernel support vector machine classifier. Therefore, when comparing these algorithms, it is unfair to automatically rule out that the faster one is the one that was programmed better due to their different solution approaches. However, these comparisons should give an idea of which algorithm obtains the best model and, in particular, which one is faster and under what circumstances, if possible.

## How to compile and execute these files
For the following, it should be noted that after executing all these programs, either by console or outputted .png/.csv files, it is expected to have obtained some evaluation metrics and execution times of all the processes performed in order to evaluate and compare them.

### How to compile the CenyML program:
1. Open the terminal and change its current working directory to the one that contains this README.md file (considering that the CenyML project files have not been changed).
2. All the ".c" programs that are contained in the current directory were made with the CenyML library. To compile them, enter the following commands into the terminal:

```console
$ make
```

**NOTE:** After these commands, several executable files with the names: "main_linearKernel.x"; "main_polynomialKernel.x"; "main_logisticKernel.x" and; "main_gaussianKernel.x" will be created, where each of them will contain the compiled program of their respectives C files but with the extension ".x".

3. Run the compiled desired program made with the CenyML library with the command in the terminal window:
```console
$ ./nameOfTheProgram.x
``` 

**NOTE:** Substitute "nameOfTheProgram" for the particular name of the compiled file you want to excecute.
For example:

```console
$ ./main_linearKernel.x
``` 

### How to compile the scikit-learn program:
1. Compile and excecute the desired file, that was made with the scitkit-learn library, with your preferred method (the way I compiled and ran it was through the Spyder IDE).

### How to compile the Dlib program made with its Python API:
1. Compile and excecute the "radialBasisFunctionKSVM_Dlib.py" file, that was made with the Dlib library, with your preferred method (the way I compiled and ran it was through the Spyder IDE).

### How to compile the Dlib program made with C++:
1. Enter the directory address through the terminal window: "/DlibCplusplus/radialBasisFunctionKSVM/main.cpp".
2. Enter the following commands into the terminal to compile the "main.cpp" file, containing the program made with the Dlib library in C++:

```console
$ make
```

**NOTE:** After these commands, an excecutable file named "main.x" will be created and will contain the program resulting from compiling the "main.cpp" file.

11. Enter the following commands into the terminal to excecute the compiled file:

```console
$ ./main.x
```

## How to interpret the results obtained
Read the messages that were displayed in the terminal where you executed the chosen programs of this folder and analyze the results obtained. Review the documentation of the databases used, which is located in several .pdf files in the directory address "../../../../databases" with respect to the directory address of this README.md file. Compare the obtained models and determine if they correspond to the mathematical equation with which the databases were generated. In addition, compare all the evaluation metrics made and the processing times obtained with each tested program in order to obtain your own conclusions.
  
# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
