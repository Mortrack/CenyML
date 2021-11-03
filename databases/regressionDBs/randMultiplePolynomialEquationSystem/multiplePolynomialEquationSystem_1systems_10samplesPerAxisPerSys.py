
# ------------------------------- #
# ----- IMPORT OF LIBRARIES ----- #
# ------------------------------- #
import random
import math


# ------------------------------------ #
# ----- MAIN PROGRAM STARTS HERE ----- #
# ------------------------------------ #
# ----- START OF: Variables to be mannually defined by the implementer ----- #
f = open("multiplePolynomialEquationSystem_1systems_10samplesPerAxisPerSys.txt","w+") # Create a new file to store data in it.
totalSystemsEvaluated = 1; # This will define the number of systems that will be sampled.
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
            currentRandomValue1 = (random.random()*2-1)*10 # We generate random values from -10 up to +10
            x2 = (sampleTime * currentSample2)
            currentIndVarData1 = (82-(1.6)*x1+(0.016)*(x1**2)-(1.28)*x2+(0.0128)*(x2**2)) + currentRandomValue1 # This is the equation that will govern the generated systems.
            f.write(format(currentId) + ";" + format(currentSystem) + ";" + format(currentIndVarData1) + ";" + format(x1) + ";" + format(x2) + "\n") # Write the new line of data into the file.
            currentId = currentId + 1 # Increase the counter of the current row Id

# ----- Close the file when the program is done inserting data into it ----- #
f.close()
