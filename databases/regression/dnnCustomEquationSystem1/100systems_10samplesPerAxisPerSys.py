
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# AUTHOR: César Miranda Meza
# COMPLETITION DATE: September 21, 2021.
# LAST UPDATE: November 08, 2021.
#
# This code is used to create a database for certain evaluation and validation
# purposes of the CenyML library.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Python version 3.9.7



# ------------------------------- #
# ----- IMPORT OF LIBRARIES ----- #
# ------------------------------- #
import random
import math
import numpy as np


# ------------------------------------ #
# ----- MAIN PROGRAM STARTS HERE ----- #
# ------------------------------------ #
# ----- START OF: Variables to be mannually defined by the implementer ----- #
f = open("100systems_10samplesPerAxisPerSys.txt","w+") # Create a new file to store data in it.
totalSystemsEvaluated = 100; # This will define the number of systems that will be sampled.
samplesGenPerAxisPerSystem = 10; # This will define the number of samples to generate for each axis of the requested sampled system.
rangeOfIndependentVariables = 100; # This will define the values to be generated for the independent variables will be ranged from 0 to the value defined in this variable.
sampleTime = rangeOfIndependentVariables / samplesGenPerAxisPerSystem # This will define the sample time in between each sample generated for each of the generated systems.
f.write("id;system_id;dependent_variable;independent_variable_1;independent_variable_2\n") # Write the header titles for each of the desired columns.
# ---- END OF: Variables to be mannually defined by the implementer ----- #

# ----- Generate the sample data and store it into the specified file ----- #
currentRandomValue1 = 0 # This variable will be used by the program to store random values
currentIndVarData1 = 0 # This variable will be used by the program to store each of the outputs of the independent variable 1.
currentId = 1 # This varible will be used by the program to store the value of the current Id to be tagged for the current sample generated.
x1 = 0 # This variable will be used to save in memory the current value of the independent variable 1
x2 = 0 # This variable will be used to save in memory the current value of the independent variable 2
# For-loop for each system to generate
for currentSystem in range(1, totalSystemsEvaluated+1):
    # For-loop for each sample to generate for each of the systems to generate
    for currentSample1 in range(1, samplesGenPerAxisPerSystem+1):
        x1 = (sampleTime * currentSample1)
        for currentSample2 in range(1, samplesGenPerAxisPerSystem+1):
            currentRandomValue1 = 0 # We will not generate a random bias value.
            x2 = (sampleTime * currentSample2)
            # LAYER 1
            Ne_1_1 = np.tanh(0 - 0.4*x1 + 0.4*x2)
            Ne_2_1 = -2.34 + 0.09*x1 + 0.77*x2; # 1st order degree
            Ne_3_1 = 1/(1+math.exp(-1.26 - 0.3*x1 + 0.5*x2)); # logistic
            Ne_4_1 = (-0.91 +0.04*x1 - 0.0037*x2)**4; # 4th order degree
            Ne_5_1 = math.exp(1.64 - 0.022*x1 - 0.022*x2); # 1st order exponential
            
            # LAYER 2
            Ne_1_2 = 0 + Ne_1_1 + 0.1*Ne_2_1 + Ne_3_1 + Ne_4_1 + 2.19*Ne_5_1; # 1st order degree
            Ne_2_2 = np.tanh(0 + Ne_1_1 - 0.1*Ne_2_1 + Ne_3_1 + 0.43*Ne_4_1 + 1.38*Ne_5_1); # Hiperbolic tangent
            Ne_3_2 = math.exp((-0.2 - 1.1*Ne_1_1 + 0.01*Ne_2_1 - 0.02*Ne_3_1 + 0.0074*Ne_4_1 + 0.017*Ne_5_1)**2); # 2nd order exponential
            Ne_4_2 = (-0.002 + 0.02*Ne_1_1 - 0.002*Ne_2_1 - 0.3*Ne_3_1 - 0.019*Ne_4_1 + 0.017*Ne_5_1)**6; # 6th order degree
            
            # LAYER 3
            Ne_1_3 = 1./(1+math.exp(1 + 0.4*Ne_1_2 - 0.9*Ne_2_2 - Ne_3_2 - 1.27*Ne_4_2)); # logistic
            
            currentIndVarData1 = Ne_1_3 + currentRandomValue1 # This is the equation that will govern the generated systems.
            f.write(format(currentId) + ";" + format(currentSystem) + ";" + format(currentIndVarData1) + ";" + format(x1) + ";" + format(x2) + "\n") # Write the new line of data into the file.
            currentId = currentId + 1 # Increase the counter of the current row Id

# ----- Close the file when the program is done inserting data into it ----- #
f.close()
