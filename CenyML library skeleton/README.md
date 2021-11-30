# /'CenyML library skeleton'
This folder contains the skeleton structure or framework that is suggested to use when creating a C program with the CenyML library.

# How to explore the CenyML project files.
The following will describe the general purpose of the folders that are located in the current directory address:

- **/CenyML_Library/**:
    - This folder will contain all the CenyML library files, along with their respective documentation (in the form of comments just before the code of each function begins).
- **/otherLibraries/**:
    - This folder will contain all the external or complementary library files with respect to the machine learning library files that the CenyML project contributed with, along with their respective documentation (in the form of comments just before the code of each function begins).

# Requirements to be able to compile this project
The following will describe all the requirements that the user has to meet in order to be able to successfully compile and excecute this CenyML framework.

## Programming languages used
The CenyML programs were compiled in C language with the compiler GCC version 10.2.0 with -O3 code optimization. In addition, the Nvidia CUDA Compiler (NVCC) in its version 11.2.142 with -O3 code optimization was used for GPU programming.

## External libraries and packages used
The following list will detail the version of the external libraries and packages that were used:
- pbPlots --> version 0.1.9.0

### How to install the external libraries and packages that are required
All the libraries used in C were developed within the framework of the CenyML project, with the exception of the pbPlots library. For that particular library, the following must be made:

1. Download the pbPlots library version 0.1.9.0 --> https://github.com/InductiveComputerScience/pbPlots/archive/refs/tags/v0.1.9.0.zip
2. Unzip the downloaded file and then enter its folder named "C".
3. Copy the files "pbPlots.c", "pbPlots.h", "supportLib.c" and "supportLib.h" and paste them in the directory folder 'CenyML library skeleton/otherLibraries/pbPlots/' (with respect to the root directory of this project). Alternatively, if you download the CenyML project files, the pbPlots library with the right version should already be there.

## Materials
This CenyML framework was successfully compiled and excecuted in a computational system with the following hardware:

- Motherboard: 1 x HUANANZHI X99Dual-F8D.
- CPU: 2 x Intel(R) Xeon(R) E5-2699V4 @ 2.10GHz.
- RAM: 1 x SAMSUNG M386A4G40EM2-CRC.
- Storage device: 1 x Samsung SSD 970 EVO plus.
- 1 x GeForce GTX 1660 SUPER (connected through Timack 20cm riser PCIe extension cable model B08BR7NB3W).
- GPU: 4 x Tesla K80 (two of the four physical GPUs are contained in a single enclosure that has one PCI-E connector. In other words, two Tesla K80 GPUs where each one of them has two GPUs inside, making a total of four).

**NOTE:** The CUDA files were excecuted only in the Tesla K80 GPUs.

# How to compile and excecute the CenyML framework
TODO: It is still pending to add this information since the framework has not yet been reviewed to be completely functional.

# Cite this project in yours!
TODO: Place APA citation text.
TODO: Place bibtex citation code.
