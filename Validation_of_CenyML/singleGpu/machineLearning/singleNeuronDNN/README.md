
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the single neuron in Deep Neural Network that was developed in the CenyML library in its single GPU version. The following describes the purpose of each program contained in this directory:

- **main_simpleLinearRegressionWithRelu.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a simple linear regression problem when selecting a ReLU activation function and using a single GPU with CUDA.
- **main_hyperbolicTangentRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a logistic regression problem when selecting a hyperbolic tangent activation function and using a single GPU with CUDA.
- **main_logisticRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a logistic regression problem when selecting a logistic activation function and using a single GPU with CUDA.
- **main_raisetoTheFirstPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a simple linear regression problem when selecting a "raise to the first power" or "identity" activation function and using a single GPU with CUDA.
- **main_raisetoTheFirstPowerRegression2.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a simple linear regression problem when selecting a "raise to the first power" or "identity" activation function and using a single GPU with CUDA. However, this program differs from "main_raisetoTheFirstPowerRegression.c" because "main_raisetoTheFirstPowerRegression2.c" increases the input data given to the artificial neuron by duplicating it a desired number of times. The purpose of this is to study the impact that a different number of samples in the input data has on the training process with respect to its performance.
- **main_raiseToTheSecondPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a second order degree regression problem when selecting a "raise to the second power" activation function and using a single GPU with CUDA.
- **main_raiseToTheThirdPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a third order degree regression problem when selecting a "raise to the third power" activation function and using a single GPU with CUDA.
- **main_raiseToTheFourthPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a fourth order degree regression problem when selecting a "raise to the fourth power" activation function and using a single GPU with CUDA.
- **main_raiseToTheFifthPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a fifth order degree regression problem when selecting a "raise to the fifth power" activation function and using a single GPU with CUDA.
- **main_raiseToTheSixthPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a sixth order degree regression problem when selecting a "raise to the sixth power" activation function and using a single GPU with CUDA.
- **main_firstOrderExponentialRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a first order degree exponential regression problem when selecting a "first order exponential" activation function and using a single GPU with CUDA.
- **main_secondOrderExponentialRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a second order degree exponential regression problem when selecting a "second order exponential" activation function and using a single GPU with CUDA.


It is important to note that the codes presented have the goal of validating the functionality of the single neuron in Deep Neural Network algorithm in its single GPU version that is provided by the CenyML library. However, the program from which the main conclusions are obtained is "main_raisetoTheFirstPowerRegression.c" because a unitary activation function is used. Therefore, that program is the least complex one and reduces the chances of making bias comparisons and conclusions. At the same time, the results obtained in that program should give the fastest results due to the fact of being the least complex one. Conclusively, these results should give an idea of how fast this algorithm performs and what are its limitations.

## How to compile and execute these files
For the following, it should be noted that after executing all these programs, either by console or outputted .png/.csv files, it is expected to have obtained some evaluation metrics and execution times of all the processes performed in order to evaluate and compare them.

### How to compile the CenyML programs:
1. Open the terminal and change its current working directory to the one that contains this README.md file (considering that the CenyML project files have not been changed).
2. All the .c programs that are contained in the current directory were all made with the CenyML library. To compile them, enter the following commands into the terminal:

```console
$ make
```

**NOTE:** After these commands, several executable files with the names: "main_simpleLinearRegressionWithRelu.x"; "main_hyperbolicTangentRegression.x"; "main_logisticRegression.x"; "main_raisetoTheFirstPowerRegression.x"; "main_raisetoTheFirstPowerRegression2.x"; "main_raiseToTheSecondPowerRegression.x"; "main_raiseToTheThirdPowerRegression.x"; "main_raiseToTheFourthPowerRegression.x"; "main_raiseToTheFifthPowerRegression.x"; "main_raiseToTheSixthPowerRegression.x"; "main_firstOrderExponentialRegression.x" and; "main_secondOrderExponentialRegression.x" will be created, where each of them will contain the compiled program of their respective C files but with the extension ".x".

3. Run the compiled desired program made with the CenyML library with the following command in the terminal window:

```console
$ ./nameOfTheProgram.x
```

**NOTE:** Substitute "nameOfTheProgram" for the particular name of the compiled file you want to excecute.
For example:

```console
$ ./main_raisetoTheFirstPowerRegression.x
``` 

## How to interpret the results obtained
Read the messages that were displayed in the terminal where you executed the chosen programs from this folder and analyze the results obtained. In addition, review the results of the evaluation metrics; the coefficients obtained and the plotted graph from the generated .csv and .png files in order to obtain your own conclusions. If more information is needed about a specific database that was used, examine their documentation which is available in several .pdf files in the directory address "../../../../databases" with respect to the directory address of this README.md file. However, note that in some C files, the data was generated in the C file itself and was not extracted from a particular database (see the preprocessing section of such C file to verify this).

# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
