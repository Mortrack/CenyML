
# Explanation of the purpose of these files and how to use them
  
## What are these files for
The files located in this directory path are all used for the validation of the single neuron in Deep Neural Network that was developed in the CenyML library. The following describes the purpose of each program contained in this directory:

- **main_simpleLinearRegressionWithRelu.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a simple linear regression problem when selecting a ReLU activation function.
- **main_hyperbolicTangentRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a logistic regression problem when selecting a hyperbolic tangent activation function.
- **main_logisticRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a logistic regression problem when selecting a logistic activation function.
- **main_raisetoTheFirstPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a simple linear regression problem when selecting a "raise to the first power" or "identity" activation function.
- **main_raiseToTheSecondPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a second order degree regression problem when selecting a "raise to the second power" activation function.
- **main_raiseToTheThirdPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a third order degree regression problem when selecting a "raise to the third power" activation function.
- **main_raiseToTheFourthPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a fourth order degree regression problem when selecting a "raise to the fourth power" activation function.
- **main_raiseToTheFifthPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a fifth order degree regression problem when selecting a "raise to the fifth power" activation function.
- **main_raiseToTheSixthPowerRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a sixth order degree regression problem when selecting a "raise to the sixth power" activation function.
- **main_firstOrderExponentialRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a first order degree exponential regression problem when selecting a "first order exponential" activation function.
- **main_secondOrderExponentialRegression.c** creates a single neuron in Deep Neural Network model with the CenyML library to solve a second order degree exponential regression problem when selecting a "second order exponential" activation function.
- **main_simpleLinearRegression_pytorch.py** creates a deep learning model with the least number of neurons possible with the pytorch library to solve a simple linear regression problem for comparative purposes with the "main_raisetoTheFirstPowerRegression.c" file.
- **main_simpleLinearRegression_scikitLearn.py** creates a deep learning model with the least number of neurons possible with the scikit-learn library to solve a simple linear regression problem for comparative purposes with the "main_raisetoTheFirstPowerRegression.c" file.
- **main_simpleLinearRegression_tensorflow.py** creates a deep learning model with the least number of neurons possible from the class "Sequential()" with the tensorflow library to solve a simple linear regression problem for comparative purposes with the "main_raisetoTheFirstPowerRegression.c" file.

It is important to note that the codes made with PyTorch, scikit-learn and Tensorflow may utilize a different mathematical approach than the one used in the CenyML library. However, with respect to the time available and my understanding of their documentation, I determined that the chosen algorithms were the most similar to the one used in the CenyML library. Therefore, when comparing these algorithms, it is unfair to automatically rule out that the faster one is the one that was programmed better due to their different solution approaches. Nonetheless, these comparisons should give an idea of which algorithm performs better or faster than the others, but under the specific circumstances tested.

## How to compile and execute these files
For the following, it should be noted that after excecuting all these programs, either by console or outputted .png/.csv files, it is expected to have obtained some evaluation metrics and excecution times of all the processes performed in order to evaluate and compare them.

### How to compile the CenyML programs:
1. Open the terminal and change its current working directory to the one that contains this README.md file (considering that the CenyML project files have not been changed).
2. All the .c programs that are contained in the current directory were all made with the CenyML library. To compile them, enter the following commands into the terminal:
```console
$ make
```
**NOTE:** After these commands, several executable files with the names: "main_simpleLinearRegressionWithRelu.x"; "main_hyperbolicTangentRegression.x"; "main_logisticRegression.x"; "main_raisetoTheFirstPowerRegression.x"; "main_raiseToTheSecondPowerRegression.x"; "main_raiseToTheThirdPowerRegression.x"; "main_raiseToTheFourthPowerRegression.x"; "main_raiseToTheFifthPowerRegression.x"; "main_raiseToTheSixthPowerRegression.x"; "main_firstOrderExponentialRegression.x" and; "main_secondOrderExponentialRegression.x" will be created, where each of them will contain the compiled program of their respectives C files but with the extension ".x".

3. Run the compiled desired program made with the CenyML library with the following command in the terminal window:
```console
$ ./nameOfTheProgram.x
```

**NOTE:** Substitute "nameOfTheProgram" for the particular name of the compiled file you want to excecute.
For example:

```console
$ ./main_raisetoTheFirstPowerRegression.x
``` 

### How to compile the PyTorch programs:
1. Compile and excecute the desired Python file, that was made with PyTorch, with your preferred method (the way I compiled and ran it was through the Spyder IDE).

### How to compile the scikit-learn programs:
1. Compile and excecute the desired Python file, that was made with scikit-learn, with your preferred method (the way I compiled and ran it was through the Spyder IDE).

### How to compile the Tensorflow programs:
1. Compile and excecute the desired Python file, that was made with Tensorflow, with your preferred method (the way I compiled and ran it was through the Spyder IDE).

## How to interpret the results obtained
Read the messages that were displayed in the terminal where you executed the chosen programs from this folder and analyze the results obtained. In addition, review the results of the evaluation metrics; the coefficients obtained and the plotted graph from the generated .csv and .png files in order to obtain your own conclusions. If more information is needed about a specific database that was used, examine their documentation which is available in several .pdf files in the directory address "../../../../databases" with respect to the directory address of this README.md file. However, note that in some C files, the data was generated in the C file itself and was not extracted from a particular database (see the preprocessing section of such C file to verify this). Finally, have in mind that the python files were compared only with respect to the "main_raisetoTheFirstPowerRegression.c" file because they solve the same type of problem.

# Permissions, conditions and limitations to use this Library  
In accordance to the Apache License 2.0 which this Library has, it can used for commercial purposes, you can distribute it, modify it and/or use it privately as long as a copy of its license is included. It is also requested to give credit to the author and to be aware that this license includes a limitation of liability and explicitly states that it does NOT provide any warranty.
