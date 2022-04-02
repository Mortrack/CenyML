# /'CenyML\_library\_skeleton'
This folder contains the skeleton structure or framework that is suggested to use when creating a C program with the CenyML library.

# How to explore the CenyML project files.
The following will describe the general purpose of the folders that are located in the current directory address:

- **/CenyML_Library/**:
    - This folder will contain all the CenyML library files, along with their respective documentation (in the form of comments, just before the beginning of the code of each function).
- **/otherLibraries/**:
    - This folder will contain all the external or complementary library files with respect to the library files that the CenyML project contributed with, along with their respective documentation (in the form of comments, just before the beginning of the code of each function).

# Requirements to be able to compile this project
The following will describe all the requirements that the user has to meet in order to be able to successfully compile and excecute this CenyML framework.

## Programming languages used
The CenyML programs were compiled in C language with the compiler GCC version 9.3.0 with -O3 code optimization. In addition, POSIX threads were used for the CPU parallelism applied and the Nvidia CUDA Compiler (NVCC) in its version 11.4 with -O3 code optimization was used for GPU programming.

## External libraries and packages used
The following list will detail the version of the external libraries and packages that were used:
- pbPlots --> version 0.1.9.0

### How to install the external libraries and packages that are required
All the libraries used in C were developed within the framework of the CenyML project, with the exception of the pbPlots library. For that particular library, the following must be made:

1. Download the pbPlots library version 0.1.9.0 --> https://github.com/InductiveComputerScience/pbPlots/archive/refs/tags/v0.1.9.0.zip
2. Unzip the downloaded file and then enter its folder named "C".
3. Copy the files "pbPlots.c", "pbPlots.h", "supportLib.c" and "supportLib.h" and paste them in the directory folder 'CenyML\_library\_skeleton/otherLibraries/pbPlots/' (with respect to the root directory of this project). Alternatively, if you download the CenyML project files, the pbPlots library with the right version should already be there.

## Materials
This CenyML framework was successfully compiled and excecuted in a computational system with the following hardware:

- Motherboard: 1 x HUANANZHI X99Dual-F8D.
- CPU: 2 x Intel(R) Xeon(R) E5-2699V4 @ 2.10GHz.
- RAM: 1 x SAMSUNG M386A4G40EM2-CRC.
- Storage device: 1 x Samsung SSD 970 EVO plus.
- 1 x GeForce GTX 1660 SUPER (connected through Timack 20cm riser PCIe extension cable model B08BR7NB3W).
- GPU: 4 x Tesla K80 (two of the four physical GPUs are contained in a single enclosure that has one PCI-E connector. In other words, two Tesla K80 GPUs where each one of them has two GPUs inside, making a total of four).

**NOTE:** The CUDA files were excecuted only in the Tesla K80 GPUs which were entirely dedicated to the CenyML project.

# How to compile and excecute the CenyML framework
1. Open the terminal and change its current working directory to the one that contains this README.md file (considering that the CenyML project files have not been changed).
2. It is expected that the main.c file located in the current directory will be used to write all the custom code that the user wants the program to do. To compile such a program, enter the following commands in the terminal:
```console
$ make
```
**NOTE:** After these commands, several compiled files with the extension ".o" will appear. These can be deleted since the one that matters is the executable file with the name: "main.x".

3. Run the program that has just been compiled with the following command in the terminal window:
```console
$ ./main.x
```
**NOTE:** If it is necessary to add an additional header file (.h) in the "main.c" file, then you must also edit the file named "Makefile" so that this header file is also considered in the compilation process that was just explained in the above steps.

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
