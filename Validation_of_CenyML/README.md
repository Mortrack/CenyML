# /'Validation\_of\_CenyML'
This folder contains all the files that were used to validate the CenyML project and that may also serve as code example files for the application of each of the functions contained in such project.

# How to explore the CenyML project files.
The following will describe the general purpose of the folders that are located in the current directory address:

- **/cpuParallel/**:
    - This folder will contain all the files that were made in order to validate and measure the performance all of the CenyML functions in their CPU parallel version.
- **/cpuSequential/**:
    - This folder will contain all the files that were made in order to validate and compare the performance all of the CenyML functions in their sequential version with respect to several other reference libraries.
- **/multiGpu/**:
    - This folder will contain all the files that were made in order to validate and measure the performance all of the CenyML functions in their multiple GPU parallel version.
- **/singleGpu/**:
    - This folder will contain all the files that were made in order to validate and measure the performance all of the CenyML functions in their single GPU parallel version.

**NOTE:** We refer by sequential to computational programs that use only 1 CPU thread to excecute them. Conversely, parallel programs use more than 1 CPU thread to execute several instructions at the same time, which is well known to give faster results than the sequential programs.

# Requirements to be able to compile any of these validation programs
The following will describe all the requirements that the user has to meet in order to be able to obtain the same results as described in the CenyML project.

## Programming languages used
The Python codes developed were compiled using the Spyder IDE v5.2.0 with Python v3.9.7. On the other hand, the CenyML programs were compiled in C language with the compiler GCC version 9.3.0 with -O3 code optimization. In addition, POSIX threads were used for CPU parallelism and the Nvidia CUDA Compiler (NVCC) in its version 11.4 with -O3 code optimization was used for GPU programming.

## External libraries and packages used
The following list will detail the version of the external libraries and packages that were used:

- For **Python**:
    - pip --> version 21.2.4
    - tensorflow --> version 2.7.0
    - scikit-learn --> version 1.0.1
    - numpy --> version 1.21.4
    - matplotlib --> version 3.4.3
    - pandas --> version 1.3.3
    - Dlib --> version 19.22
    - statsmodels --> version 0.13.1
    - spyder --> 5.2.0
    - cmake --> version 3.16.3
- For **C**:
    - pbPlots --> version 0.1.9.0

### How to install the external libraries and packages that are required
To install the libraries for Python, make sure you have installed anaconda and then type in the following commands in any terminal window:

```console
$ conda update conda
$ conda update anaconda
$ conda create --name py3_9_7 python=3.9.7
$ conda activate py3_9_7
$ apt update
$ sudo apt install cmake
$ conda install -c conda-forge dlib==19.22
$ pip install scikit-learn==1.0.1
$ pip install matplotlib==3.4.3
$ pip install pandas==1.3.3
$ pip install spyder==5.2.0
$ pip install tensorflow==2.7.0
$ pip install numpy==1.21.4
$ pip install statsmodels==0.13.1
$ spyder
``` 


**NOTE:** If you get the error "Could not open lock file /var/lib/apt/lists/lock - open (13: Permission denied)" while trying to execute the command "apt update", then to fix this what I did was to type the following commands on the terminal window in which the error happened:

```console
$ sudo fuser -vki /var/lib/dpkg/lock-frontend
$ sudo rm -f /var/lib/dpkg/lock-frontend
$ sudo dpkg --configure -a
$ sudo apt autoremove
``` 


With the last command ($ spyder), the Spyder IDE will be opened and there you will have to open the desired Python file to then compile and run it.

Moreover, all the libraries used in C were developed as part of the CenyML project, with the exception of the pbPlots library. For that particular library, the following must be made:

1. Download the pbPlots library version 0.1.9.0 --> https://github.com/InductiveComputerScience/pbPlots/archive/refs/tags/v0.1.9.0.zip
2. Unzip the downloaded file and then enter its folder named "C".
3. Copy the files "pbPlots.c", "pbPlots.h", "supportLib.c" and "supportLib.h" and paste them in the directory folder 'CenyML library skeleton/otherLibraries/pbPlots/' (with respect to the root directory of this project). Alternatively, if you download the CenyML project files, the pbPlots library with the right version should already be there.

## Materials
All the algorithms made for this thesis product, were executed on a dedicated computer system with Ubuntu OS v20.04.3 LTS and the following hardware:

- Motherboard: 1 x HUANANZHI X99Dual-F8D.
- CPU: 2 x Intel(R) Xeon(R) E5-2699V4 @ 2.10GHz.
- RAM: 1 x SAMSUNG M386A4G40EM2-CRC.
- Storage device: 1 x Samsung SSD 970 EVO plus.
- 1 x GeForce GTX 1660 SUPER (connected through Timack 20cm riser PCIe extension cable model B08BR7NB3W).
- GPU: 4 x Tesla K80 (two of the four physical GPUs are contained in a single enclosure that has one PCI-E connector. In other words, two Tesla K80 GPUs where each one of them has two GPUs inside, making a total of four).

**NOTE:** The CUDA files were executed only in the Tesla K80 GPUs which were entirely dedicated to the CenyML project.

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
