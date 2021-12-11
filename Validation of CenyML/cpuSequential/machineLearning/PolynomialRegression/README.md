
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation and evaluation of the polynomial regression that was developed in the CenyML library. The following describes the purpose of each program contained in this directory:

- **main.c** creates a polynomial regression model with the CenyML library to solve a 4th order degree polynomial system.

It is important to note that it was desired to compare this algorithm with an existing alternative. However, with respect to the available time I had to do research, I was unable to find any machine learning library that had an algorithm like the one mathematically formulated in the CenyML library. However, the only library I did found that provided enough tools to allow me to create a similar version was scikit-learn. This is because after several tests, I realized that the multiple linear regression of scikit-learn is the same as the one formulated in the master thesis of César Miranda Meza called "Machine learning library to support applications with embedded systems and parallel computing". Now, since that method is key to create a polynomial regression as in the CenyML library, I could easily implement a few lines of code to reformulate the input data to obtain the desired equivalent method, where I could also use the scikit-learn method called "PolynomialFeatures()". However, I decided to discard the comparison to avoid giving biased results, since scikit-learn does not explicitly provide the multiple polynomial regression and because I could easily make biased comparisons by creating this method myself in/with another library.

## How to compile and execute these files
For the following, it should be noted that after executing all these programs, either by console or outputted .png/.csv files, it is expected to have obtained some evaluation metrics and execution times of all the processes performed in order to evaluate them.

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

## How to interpret the results obtained
Read the messages that were displayed in the terminal where you executed the program of this folder and analyze the results obtained. Review the documentation of the databases used, which is located in several .pdf files in the directory address "../../../../databases" with respect to the directory address of this README.md file. Compare the obtained model and determine if it corresponds to the mathematical equation with which the databases were generated. In addition, compare all the evaluation metrics made and the processing times obtained in order to obtain your own conclusions.

# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
