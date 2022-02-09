#ifndef CUDA_CHECK_H
#define CUDA_CHECK_H

/**
* The "CHECK()" function is used so that you can know when there
* is an error when sending a command/instruction to the GPU. To
* use this, the only thing you have to do is to call this function
* and then put on its argument the CUDA function you want to use
* to send a command/instruction to the GPU and, if there gets to
* be an error, this function will display on the terminal the
* error details. This is highly recommended because since the GPU
* communicates asynchronously, the CPU has no other way of knowing
* if something went wrong with the GPU.
*
* @param call void - This argument must contain a CUDA function
*                    within when you call this function.
*
* @return NULL
*
* @author John Cheng, Max Grossman & Ty McKercher
* CREATION DATE: year 2014
* LAST UPDATE: N/A
*/
#define CHECK(call)                                                         \
{                                                                           \
    const cudaError_t error = call;                                         \
    if (error != cudaSuccess)                                               \
    {                                                                       \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                       \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));  \
        exit(1);                                                            \
    }                                                                       \
}

#endif

