/*


CUDA built-in variables: gridDim, blockDim, blockIdx, and threadIdx
    - Their values are preinitialized by the CUDA runtime systems and can be referenced 
    in the kernel function. 


A grid is a 3D array of blocks1 and each block is a 3D array of threads.



Transparent scalability: The ability to execute the same application code on hardware
with a different number of execution resources.
    - SPMD: single program, multiple data


Streaming multiprocessors (SMs)
    - execution resources are organized into SMs
    - many blocks in 1 SM
    - Up to 8 blocks to each SM as resource allows
    - up to 1,536 threads can be assigned to each SM
    - SM maintains thread & block id #s
    - SM manages thread execution
    - 1 GPU (CUDA device) has 1+ SMs

Thread scheduling
    - a block is assigned to a SM, is divided into many 32-thread units called warps
    - a warp is a collection of threads
    - Warps are scheduling units in SM
    - Blocks are partitioned into warps for thread scheduling.
    - one instruction is fetched and executed for all threads in a warp
    - At any time, 1 or 2 of the warps is executed by a SM
    - streaming processors(SPs) inside a SM, is the actual hardware that execute instructions
    - The selection of ready warps can let 
        - CUDA processors efficiently execute long-latency operations such as global memory accesses.

    - Branching divergence
        - Threads within a single warp take different paths
        - Avoid Branching divergence such as:
        - If (threadIdx.x > 2) { }
            - thread 0, 1, 2 in warp#1 have different path with thread 3, 4, .. 31 in warp#1


Latency tolerance or latency hiding
    - filling the latency time of operations with work from other threads


//  barrier synchronization function
__syncthreads();    // all threads in a block will be held at the calling location
                    // until every thread in the block reaches the location


*/

int vecAdd(float *A, float *B, float *C, int thread_need)
{
    // A_d, B_d, C_d allocations and copies omitted

    // using struct to specify 3D dimension x, y, z
    dim3 DimGrid(ceil(thread_need / 256.0), 1, 1);
    dim3 DimBlock(256, 1, 1); // 256 threads in x-direction

    // if (thread_need % 256)
    //     DimGrid.x++; // Run ceil(n/256) blocks of 256 threads each

    vecAddKernel<<<DimGrid, DimBlock>>>(A_d, B_d, C_d, n);

    // or shortcut, default set y, z to 1
    // vecAddKernel<<< ceil(n / 256.0), 256 >>>(A_d, B_d, C_d, n);
}

// current CUDA C compiler leaves the work of
// “flatten” a dynamically allocated 2D array into an equivalent 1D array
// such translation to the programmers due to lack of dimensional information.
__global__ void PictureKernell(float *d_Pin, float *d_Pout, int n, int m)
{
    // Calculate the row # of the d_Pin and d_Pout element to process
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column # of the d_Pin and d_Pout element to process
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    // mapping of 1 threads to each data set
    // each thread computes one element of d_Pout if in range
    if ((Row < m) && (Col < n))
    {
        // d_Pout[row][col] = 2 x d_Pin[row][col]
        d_Pout[Row * n + Col] = 2 * d_Pin[Row * n + Col];
    }
}

#define BLOCK_WIDTH 16 // 16 threads in x-direction of block

// Setup the execution configuration
int NumBlocks = Width / BLOCK_WIDTH;

if (Width % BLOCK_WIDTH)
    NumBlocks++;

dim3 dimGrid(NumBlocks, NumbBlocks);
dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

// Launch the device computation threads!
matrixMulKernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, Width);

// (Matrix M) x (Matrix N) = (Matrix P)
// Each thread calculates one element of P
__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int Width)
{
    // Calculate the row index of the d_P element and d_M
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    // Calculate the column index of d_P and d_N
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < Width) && (Col < Width))
    {
        float Pvalue = 0;
        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < Width; ++k)
        {
            Pvalue += d_M[Row * Width + k] * d_N[k * Width + Col];
        }

        //
        d_P[Row * Width + Col] = Pvalue;
    }
}

// Matrix multiplication on the (CPU) host in double precision
void MatrixMulOnHost(float *M, float *N, float *P, int Width)
{
    for (int i = 0; i < Width; ++i)
        for (int j = 0; j < Width; ++j)
        {
            double sum = 0;
            for (int k = 0; k < Width; ++k)
            {
                double a = M[i * Width + k];
                double b = N[k * Width + j];
                sum += a * b;
            }

            P[i * Width + j] = sum;
        }
}
