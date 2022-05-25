# /'STM32F446RE\_Examples'
This folder contains some simple application examples of using the CenyML library in an STM32F446RE ARM microprocessor to train some machine learning models; and in some cases to make predictions and/or apply a validation code for the trained model.

# How to explore the CenyML project files.
The following will describe the general purpose of the folders that are located in the current directory address:

- **/SimpleLinearRegression/**:
    - This folder will contain all the files that were coded in order to apply a simple linear regression model in the STM32F446RE microprocessor with the CenyML library.
- **/SingleNeuronInDNN/**:
    - This folder will contain all the files that were coded in order to solve a linear regression system with a single neuron in a Deep Neural Network model in the STM32F446RE microprocessor with the CenyML library.

# How these projects were configured, compiled and executed

## How to install the STM32CubeIDE application v1.8.0
First, make sure you have the STM32CubeIDE application installed already and, if not, follow these next steps to install it in a computer with the Linux-Ubuntu Operative System:

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

## Steps followed to create a new project with the STM32CubeIDE editor

1. Create a new folder in this path and give it the name you desire to identify the project to be created.
2. Open the STM32CubeIDE launcher.
3. In the STM32CubeIDE launcher window, define the workspace directory address to be fixed inside the folder that was created in step 1 (IMPORTANT: Do not check the box for the label "Use this as the default and do not ask again").
4. In the STM32CubeIDE launcher window, click the "Launch" button. This will open the STM32CubeIDE project editor and will create a hidden folder (named ".metadata") that will allow your STM32CubeIDE editor to identify that the contents of this folder correspond to an STM32CubeIDE project.
5. In the "Project Explorer" section, click the hyperlink "Create a New STM32 project", which will open a new window called "STM32 Project".
6. In the "Part Number" scroll-down bar of the "STM32 Project" window, type in the model of the embedded system of interest, which in this case is the "STM32F446RE".
7. Under the "MCUs/MPUs" list of the "STM32 Project" window, select the matching part number with respect to the one typed in step 6 and then click the "Next >" button.
8. In the window that should have poped, define the desired project name (for simplicity and consistency purposes, I am defining the name of "Project" for all STM32CubeIDE projects).
9. Make sure to enable the checkbox labeled as "Use default location" and select the options of: "C" for the "Targeted Language" section; "Executable" for the "Targeted Binary Type" section; and "STM32Cube" for the "Targeted Project Type" section. Finally, click the "Finish" button.
10. Click the "Yes" button in the "Open Associated Perspective" window that should have appeared after completing step 9.

## Steps followed to configure the device with the STM32CubeIDE editor (Pinout & Configuration, Clock Configuration, Project Manager and Tools sections)

1. In the section "Pinout & Configuration" of the "Device Configuration Tool" window, define the following peripherals manually by literally clicking the microprocessor image in the "Pinout view" section:

	- PC13 --> RTC_AF1.
	- PC14 --> RCC_OSC32_IN.
	- PC15 --> RCC_OSC32_OUT.
	- PH0 --> RCC_OSC_IN.
	- PH1 --> RCC_OSC_OUT.

2. Click the menu dropdown "System Core" that is located in the "Categories" subsection of the "Pinout & Configuration section", then select "RCC" and select the option "Crystal/Ceramic Resonator" for both the "High Speed Clock (HSE)" and "Low Speed Clock (LSE)" options.
3. Click the menu dropdown "Timers" that is located in the "Categories" subsection of the "Pinout & Configuration section", then select "RTC" and check both the "Activate Clock Source" and "Activate Calendar" boxes; and choose the option "Timestamp Routed to AF1" in the "Timestamp" option.
4. In the section "Clock Configuration" of the "Device Configuration Tool" window, define the value of "8" in the "Input frequency" box that has the range of "4-26 MHz"; choose the "HSE" option in "PLL Source Mux"; choose the "PLLCLK" option in "System Clock Mux"; and the value of "180" in the "HCLK (MHz)" box that has the label "180 MHz max" and then click the "Enter" button of your keyboard right away after having typed the "180" value, so that the STM32CubeIDE editor iterates to configure the remaining values automatically.
5. Click the "Save (Ctrl+S)" button in the STM32CubeIDE editor and, right away, a "Question" window should pop with the content of "Do you want generate Code?". In such window that popped, click the "Yes" button so that the STM32CubeIDE editor generates the corresponding code for the project that considers all the device configurations that have been made so far.
6. Another window called "Open Associated Perspective?" should pop with the message "This action can be associated with C/C++ perspective. Do you want to open this perspective now?". Under such window, check the box with the label "Remember my decision" and then click the "Yes" button.


## How to configure the compiler settings and the STM32CubeIDE editor to see printf messages in console and to execute the program

1. Inside the "main.c" file (located in the directory address "/Core/Src" with respect to the project folder), identify the following commented section:

```C
/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */
```

2. In between the "/* USER CODE BEGIN 0 */" and "/* USER CODE END 0 */" comments that were identified in the step 1, add the following code that is required to forward all the "printf()" commands to the ITM:

```C
/*
 * The _write() function is used to forward all the printf commands to the ITM,
 * which is the STM32 IDE component that manages the data console of such IDE.
 * However, in order for you to see any printed messaged in the SWV ITM Data
 * Console tool, all your printf strings will have to end with a new line (\n).
 *
 * return int len
 *
 * @author Controllers Tech
 * CREATION DATE: JANUARY 07, 2022.
 * LAST UPDATE: N/A/
 */
int _write(int file, char *ptr, int len) {
	for (int i=0; i<len; i++) {
		ITM_SendChar((*ptr++));
	}
	return len;
}
```

3. Connect the STM32 development board to your computer and then click on the "Debug Project" button, where the "Edit Configuration" window should have opened after that.

4. Click the "Debugger" tab of the "Edit Configuration" window.
	
	1. Click the "Debug probe" menu dropdown and select "ST-LINK (ST-LINK GDB Server)".
	
	2. In the "Serial Wire Viewer" section, check the "Enable" box.
	
	3. In the "Core Clock (MHz)" box, rewrite the value to "180" because the STM32F446RE microprocessor is expected to be configured to 180MHz core clock frequency if all the steps have been followed so far.
	
	4. Click the "Apply" button.

5. Click the "Main" tab of the "Edit Configuration" window.
	1. Click in the hyperlink "Build Configuration" to open the "Properties for Project" window.
	2. In the "Properties for Project" window, click the dropdown arrow of the menu text "C/C++ Build" and then select the "Settings" options.
	3. In the content of such window, select the "MCU Settings" and then check the option "Use float with printf from newlib-nano (-u \_printf\_float)".
	4. In the content of such window, selected the "Optimizations" option of the "MCU GCC Compiler" dropdown list. Then, select the option "Optimize most (-O3)" in "Optimization level".
	5. Click the "Apply and Close" button of the "Properties for Project" window.

6. Click the "OK" button to enter in debug mode (a window called "Confirm Perspective Switch" should pop after clicking this button, where it is suggested to check the box labeled as "Remember my decision" to then click the "Switch" button).

7. Click the "SWV ITM Data Console" option from the "Window --> Show View --> SWV" dropdown menu.
	1. Click the "Configure trace" button from the console: "SWV ITM Data Console".
	2. Check the "Enable" option of any comparator and assign it the "Var/Addr", the value of "0x0".
	3. In the "ITM Stimulus Ports" section, check the "Enable port:" with identifier "0" and then click the "Ok" button.
	4. Click the "Start Trace" button to activate its functionality.
	
8. After this point, you have successfully configured your project so that your STM32 can work as requiered in these example files. Therefore, you may now click the button "Terminate (Ctrl+F2)" to exit the debug mode. Subsequently, you can now add all the code you want in the "main.c" file of the project (located in the directory address "/Core/Src" with respect to the project folder) and even use as many "printf()" functions as you desire.
	1. Once you have finished adding the desired code into the "main.c" file, click the "Build All (Ctrl+B)" button to compile all your code. If there are no code errors, you should have received a successfull build message in the "Console" window right after having completed step 1 (e.g., "16:00:21 Build Finished. 0 errors, 0 warnings. (took 509ms)"). Make sure you sucessfully compile your code before going to further steps.
	2. Click the "Debug Project" button to enter in debug mode.
	3. Once being in debug mode, click the "SWV ITM Data Console" tab to display its corresponding window, since it is here where all the messages from any "printf()" commands will be shown.
	4. And that's it!. Either click the "Resume (F8)" button to execute the compiled program from your STM32 board; execute your program step by step; or execute your program up to a certain line of code, whatever you desire.

Have fun!.

# Cite this project in yours!

## APA citation
Paper in progress.

## BibTeX citation

```$bibtex
Paper in progress.
```
