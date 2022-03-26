# CenyML
Inspired in the "Mortrack_ML_Library", CenyML provides a Machine Learning library and API programmed in C that was created by César Miranda Meza durin his master's degree at IPN (CITEDI) and was completed on Tuesday, February 8, 2022. This software contributes with a complete set of machine learning tools that have been scientifically validated and that have the philosophy of increasing performance to the maximum and having transparent output models and documentation. In addition, the model transparency of the CenyML algorithms will allow the users to train and generate models to then use them even withot the strict need of this particular library. Furthermore, the user has the option of using CenyML through its software library or in its API format, depending on his needs. For example, the API could be used to train a certain model on a remote high performance server. Subsequently, the generated model could be implemented in a low cost microcontroller/microprocessor or embedded system through its library format (e.g., Arduino or literally any microcontroller you can think of that has enough memory to store either the whole library or the particular algorithm required from it).

- Latest public stable version of the CenyML software: 0.151.111.30 (released on March 26, 2022 in master branch)
- Latest public stable version of the CenyML Library: 0.151.111.30 (released on March 26, 2022 in master_library branch)
- Latest public stable version of the CenyML API: 0.0.0.0 (released on September 21, 2021 in master_api branch)

**NOTE:** Versions are assigned according to the following list, where each one of them describes an independent numeric value of the published version, out of a total of 4 values X.X.X.X:

1. "Number of core phylosophies applied to the software" (having this digit as zero will mean that the software has not yet have been completed and released to the public for the first time).
2. "Number of new functionalities added with respect to the current core phylosophy".
3. "Number of improvements made with respect to the current core phylosophy".
4. "Number of errors/bugs that have been fixed with respect to the current core phylosophy"

# How to explore the CenyML project files.
The following will describe the general purpose of the folders that are located in the current directory address:

- **/'ArduinoExample'**:
    - This folder contains application examples of using the CenyML library in an Arduino UNO to train machine learning models; make predictions and; in some cases, apply a validation code for them.
- **/'CenyML\_library\_skeleton'**:
    - This folder will provide a skeleton of the structure or framework that is suggested for the user to use for the development of any desired program that uses the CenyML C library in it.
    - There, more information about the CenyML library documentation will be available, which will explain how to use the CenyML library and how to compile it in the skeleton project.
    - The user will be able to find more information related to any external libraries that have been implemented in this skeleton folder in order to complement the CenyML functionalities.
- **/databases**:
    - More information about the databases generated for the validation of the CenyML project and the databases themselves can be found there.
- **/'STM32F446RE\_Examples'**:
    - This folder contains application examples of using the CenyML library in an STM32F446RE development board to train machine learning models; make predictions and; in some cases, apply a validation code for them.
- **/'Validation\_of\_CenyML'**:
    - Here, all the validation files that have been developed and any additional information about them can be found. In addition, please note that these validation files also contribute to serve as code examples of how to apply each of the functions provided by the CenyML project.

# What is the purpose of each branch created in this project?
The following list will describe the purpose for which each branch of this project is used:

- **master**:
    This branch is used to contain the latest stable versions for both the CenyML library and its API. This will also include all the corresponding validation, documentation and example files.
- **master_library**:
    This branch is used to contain the latest stable version for only the CenyML library. This will also include all the corresponding validation, documentation and example files.
- **master_api**:
    This branch is used to contain the latest stable version for only the CenyML API. This will also include all the corresponding validation, documentation and example files.
- **dev_library**:
    This branch is used to contain all the latest development files for the CenyML library only. However, its use is not recommended because it is highly possible that one or more functionalities will not work properly in this branch.
- **dev_api**:
    This branch is used to contain all the latest development files for the CenyML API only. However, its use is not recommended because it is highly possible that one or more functionalities will not work properly in this branch.

# Cite this project in yours!
TODO: Place APA citation text.
TODO: Place bibtex citation code.
