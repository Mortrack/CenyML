# /'STM32F446RE\_Examples'
This folder contains some simple application examples of using the CenyML library in an STM32F446RE ARM microprocessor to train some machine learning models; make predictions and; in some cases, apply a validation code for them.

# How to explore the CenyML project files.
The following will describe the general purpose of the folders that are located in the current directory address:

- **/SimpleLinearRegression/**:
    - This folder will contain all the files that were coded in order to apply a simple linear regression model in the STM32F446RE microprocessor with the CenyML library.
- **/SingleNeuronInDNN/**:
    - This folder will contain all the files that were coded in order to solve a linear regression system with a single neuron in a Deep Neural Network model in the STM32F446RE microprocessor with the CenyML library.

# How to excecute this program

## How to install the STM32CubeIDE application v1.8.0
First, make sure you have the STM32CubeIDE application installed already and, if not, follow these next steps:

1. Go to the website https://www.st.com/en/development-tools/stm32cubeide.html and download the app with the part number labeled as "STM32CubeIDE-DEB" in its 1.8.0 version.

2. Open a terminal window in the directory at which the dowloaded file is located and then type in the following command:

```console
$ unzip en.st-stm32cubeide_1.8.0_11526_20211125_0815_amd64.deb_bundle.sh_v1.8.0.zip
```

3. Enter the following command in the same terminal window:

```console
$ sudo apt install unzip
```

4. Enter the following command in the same terminal window:

```console
$ sudo sh ./st-stm32cubeide_1.8.0_11526_20211125_0815_amd64.deb_bundle.sh
```

# How to configure the STM32CubeIDE in order to see printf messages in console and to execute the program

1. Open the desired STM32 example with the STM32CubeIDE.

2. Click the "build" button to compile the project.

3. Click on the "Debug configurations" button.

4. Click the "Debugger" tab.
	
	1. Click the "Debug probe" menu dropdown and select "ST-LINK (ST-LINK GDB Server)".
	
	2. In the "Serial Wire Viewer" section, check the "Enable" box.
	
	3. In the "Core Clock (MHz)" box, rewrite the value to "180" because the STM32F446RE microprocessor is expected to be configured to 180MHz core clock frequency in these examples.
	
	4. Click the "Apply" button.

5. Click the "Main" tab.
	1. Click in the hyperlink "Build Configuration", in the left list of the window that should have appeared.
	2. Click the dropdown arrow of the menu text "C/C++ Build" and then select the "Settings" options.
	3. In the content of such window, select the "MCU Settings" and then check the option "Use float with printf from newlib-nano (-u \_printf\_float)".
	4. In the content of such window, selected the "Optimizations" option of the "MCU GCC Compiler" dropdown list. Then, select the option "Optimize most (-O3)" in "Optimization level".

6. Click the "Apply" button and close the windows that opened for this configuration settings.

7. Click the "Debug" button to enter in debug mode.

8. Click the "SWV ITM Data Console" option from the "Window --> Show View --> SWV" dropdown menu.
	1. Click the "Configure trace" button from the console: "SWV ITM Data Console".
	2. Check the "Enable" option of any comparator and assign it the "Var/Addr", the value of "0x0".
	3. In the "ITM Stimulus Ports" section, check the "Enable port:" with identifier "0" and then click the "Ok" button.
	4. Click the "Start Trace" button to activate its functionality.
	
9. Finally, click the "Resume" button and your STM32F446RE microprocessor will execute the example application you chose and it will show all the "printf" messages in the console: "SWV ITM Data Console".

Have fun!.

# Cite this project in yours!

## APA citation
 C. Miranda, “Machine learning library to support applications with embedded systems and parallel computing” Master’s thesis, Centro de Investigación y Desarrollo de Tecnología Digital (CITEDI) of the Instituto Politécnico Nacional (IPN), Tijuana, B.C., Mexico, 2022.

## BibTeX citation
```$bibtex
@MastersThesis{cesarMirandaMeza_mastersThesis,
author = {César Miranda},
title  = {Machine learning library to support applications with embedded systems and parallel computing},
school = {Centro de Investigación y Desarrollo de Tecnología Digital (CITEDI) of the Instituto Politécnico Nacional (IPN)},
address = {Tijuana, B.C., Mexico},
year   = {2022}
}
```

**NOTE:** The URL "https://bit.ly/3iW5t9Z" links to the repository of the IPN institute where the thesis "Machine learning library to support applications with embedded systems and parallel computing" will be available in the near future. The reason this thesis is not available there yet is because this thesis has just been completed and some administrative processes are required for this repository to make it available to the public.
