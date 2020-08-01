/*

CUDA Memories


global memory, implemented with dynamic random access memory (DRAM) not faster enough than 
cache memory (shared memory in chip)



for (int k 5 0; k , Width; 11k)
    Pvalue += d_M[RowWidth 1 k] * d_N[kWidth 1 Col];    // 1 global memory access fetches a d_M[] 
                                                        // and 1 fetches a d_N[] element.   
                                                        // 1 floating-point multiplication
                                                        // and 1 floating-point addition.
                                                        // floating-point calculation : global memory access
                                                        // is 1:1
                                                        // is compute to global memory access (CGMA) ratio

Registers
    - Read/write
    ~ 1 cycles access


Global memory (is DRAMs) 
    - bandwidth of 200 GB/s
    - peak single-precision performance is 1,500 GFLOPS
    - 4 bytes each single-precision floating-point value
    - then 200/4 = 50 GFLOPS "single-precision operands per second"
    - (CGMA) ratio = 50 : 1500 = 3.3%
    ~ 500 cycles access


Constant memory
    - short-latency
    - high-bandwidth
    - Read-only access by the device when all threads simultaneously access the same location
    - cache in L1 caches
    ~ 5 cycles access


Shared Memory (Scratchpad)
    - like "L1, 2,3 cache" in CPU
    - but doesn't need to hold a copy of data that is also in main memory (like cache in CPU)
    - used in the kernel source code
    - one in each SM
    - higher speed than Global Mem (like DRAM using in CPU)
    - Scope: thread block
    - Lifetime: thread block
    - Read/write
    ~ 5 cycles access


Unified Memory
    - a single memory address space, accessible from any processor (CPU or GPU) in a system
    - calls to cudaMallocManaged(), assign mem to a pointer accessible from any processor

        cudaError_t cudaMallocManaged(void** ptr, size_t size);

    - the CUDA system software/hardware takes care of migrating memory pages to the memory 
    of the accessing processor
    - with virtual memory page faulting and migration (on-demand page migration)
    - system can migrate pages on demand from anywhere to the GPU's memory for efficient processing
    - page faulting means that only the pages, kernel to accesses, need to be migrated
        - memory migration overhead 

    - Pre-Pascal GPU
        - potentially migration overhead on each kernel launch (Host To Device, Device To Host )
            - must migrate all pages to host memory or to another GPU 
    - Pascal GPU
        - No migration overhead
            - page faulting, then Page Migration Engine migrates the pages to the device


CUDA Variable Type Qualifiers:
Variable Declaration                    | Memory      Scope       Lifetime
----------------------------------------+-------------------------------------
                        int var;        | Register    Thread      Kernel          "in reg"
                        int Var[]       | Local       Thread      Kernel          "in Local Mem"
__device__ __shared__   int SharedVar;  | Shared      Block       Kernel
__device__              int GlobalVar;  | Global      Grid        Application
__device__ __constant__ int ConstVar;   | Constant    Grid        Application     "declare outside kernel fnc"



Tiles
- partition the data into subsets
- each tile fits into the shared memory
- requires the threads to have a similar execution schedule
    - use barrier synchronization to keep threads to form the “carpool” group
    - so that they follow approximately the same execution timing
- size of tiles need to fit into the shared memory



dev_prop.regsPerBlock;              // # of registers available in each SM
regsPerBlock / maxThreadsPerBlock   // # of reg per thread can use


*/

// Just using Global Mem
__global__ void MatrixMulKernel(float *M, float *N, float *P, int Width)
{
    // Calculate the row index of the P element and M
    int Row = blockIdx.y * blockDim.y + threadIdx.y; // "Row" in reg

    // Calculate the column index of P and N
    int Col = blockIdx.x * blockDim.x + threadIdx.x; // "Col" in reg

    if ((Row < Width) && (Col < Width))
    {
        float Pvalue = 0;

        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < Width; ++k)
        {
            Pvalue += M[Row * Width + k] * N[k * Width + Col];
        }

        P[Row * Width + Col] = Pvalue;
    }
}

void blurKernel(unsigned char *in, unsigned char *out, int w, int h)
{
    // declare a shared Mem var
    __shared__ float ds_in[TILE_WIDTH][TILE_WIDTH];
}

#define TILE_WIDTH 16

// Tiled Matrix Multiplication
// using Tiling
// reduction of global mem access is by a factor of N for N x N elements.
__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int Width)
{
    // Created for each block
    // all threads of a block can access to the same Mds and Nds
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // for M matrix, device_shared mem
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    //
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x; // within TILE_WIDTH
    int ty = threadIdx.y; // within TILE_WIDTH

    // Identify the row and column of the d_P element to work on
    int Col = bx * TILE_WIDTH + tx;
    int Row = by * TILE_WIDTH + ty;
    float Pvalue = 0;

    // Loop over the d_M and d_N tiles required to compute d_P element
    // calculation is performed in (Matrix width / TILE_WIDTH) phases, # of this outer loop
    // (for a 4 x 4 matrix, 2x2 tile width, need 4/2 = 2 phases)
    // each phase update the shared Mem
    for (int m = 0; m < Width / TILE_WIDTH; ++m) // "m" is number of phases
    {
        // Each phase focuses on a small subset of the input matrix elements
        // This behavior is called locality

        // Coolaborative loading of d_M and d_N tiles into shared memory
        // every thread in a block to
        // load one M element and one N element into the shared memory
        // (m * TILE_WIDTH + tx) --- offset in Matrix M
        Mds[ty][tx] = d_M[Row * Width + (m * TILE_WIDTH + tx)]; // M is the first operand in matrix

        // (m * TILE_WIDTH + ty) --- row# in Matrix N
        Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col]; // N is loading vertically

        // ensure the completed shared Mem is loaded for this tile, by each thread in a block, is ready
        __syncthreads();

        // performs one phase of dot product
        if (Row < Width && Col < Width)
        {
            for (int k = 0; k < TILE_WIDTH; ++k)
            {
                Pvalue += Mds[ty][k] * Nds[k][tx];
            }
        }
        // ensures that all threads have finished using the shared memory
        __syncthreads();

        // then move to next phase
    }

    // P is within matrix size
    if (Row < Width && Col < Width)
        d_P[Row * Width + Col] = Pvalue;
}

// with boundary check on matrix width with tile width
for (int m = 0; m < (Width - 1) / TILE_WIDTH + 1; ++m)
{
    if (Row < Width && m * TILE_WIDTH + tx < Width)
    {
        ds_M[ty][tx] = M[Row * Width + m * TILE_WIDTH + tx];
    }
    else
    {
        ds_M[ty][tx] = 0.0;
    }

    if (m * TILE_WIDTH + ty < Width && Col < Width)
    {
        ds_N[ty][tx] = N[(m * TILE_WIDTH + ty) * Width + Col];
    }
    else
    {
        ds_N[ty][tx] = 0.0;
    }
    __syncthreads();
}