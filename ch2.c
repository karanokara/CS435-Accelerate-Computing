/*
Data Parallelism and CUDA C

CPU
    - Latency Oriented Cores
    - Less ALU (2 ~ 4 ~ 8)
        - Reduced operation latency
    - Large caches
        – Convert long latency memory
        - accesses to short latency cache accesses
    – Sophisticated control
        – Branch prediction for reduced branch latency
        – Data forwarding for reduced data latency
    - For sequential part of code where latency matters
    - Mem called Host Memory

    
GPU
    - Throughput Oriented Cores
    - Lots of ALUs
        - Many, long latency but heavily pipelined for high throughput
    - Small caches
        - To boost memory throughput
    - Simple control
        – No branch prediction
        – No data forwarding
    - massive number of threads
        - Threading logic
        - Thread state 
    - For parallel parts code where throughput wins
    - Mem called Global memory


CUDA (Compute Unified Device Architecture) C
    - an extension (API) to the popular C programming language

For CUDA Computing System consist of
    1. Host: CPU (such as Intel architecture)
    2. Devices: 1+ devices, CUDA device is GPU


Data parallelism
    - Independent evaluation on different piece of data can be done concurrently

CUDA program = C source + CUDA keyword
             = (host code) + (device code "called kernel" )

    - A kernel function is executed by all threads during a parallel phase
    - CUDA program is SPMD (single program, multiple data)
    - when launch a kernel
        - generate a grid (an array of thread blocks)           "only 1 grid??"
        - Grid (is device), has N blocks, one block has N threads, one thread has N registers
        - Each grid (device) has 1 Global Memory
        - each thread block contain up to 1024 threads (should be multiples of 32 for efficiency)
        - all thread run the same kernel code
        - Threads in different blocks do not interact

    - Need be compiled by CUDA C compler: nvcc (NVIDIA C Compiler)
    - To compile:

    nvcc myfile.cu

    - To run:

    ./a.out




                                Executed                    Only callable
                                on the:                     from the:
----------------------------------------------------------------------------
__global__ void KernelFunc()    device                      host
__device__ float DeviceFunc()   device                      device
__host__ float HostFunc()       host                        host



*/