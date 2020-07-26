/*

Parallel Patterns: Convolution
    - an important parallel computation pattern 
    - used in signal processing, digital recording, image processing, 
    video processing, computer vision
    - as a filter, transforms signals and pixels into more desirable values
    - such as
        - Gaussian filters
        - filters smooth


    - technique using
        - tiling
        - cache memory


    - how to apply?
        - convolution is an array operations
         output = weight sum of a collection of neighboring input elements
        - weighted sum calculation using "Mask Array" (convolution kernel) (convolution masks)
            - elements of "Mask Array" called "Mask Coefficients"
            - "Mask Array" using Constant memory
        - size of the mask tends to be an odd number in symmetry like: 1 2 3 2 1
        
        Input (N):                      output (P):      
                 i                                i  
            1 2 (3) 4 5 6 7                 __ __ 57 __ __ __ __ __ __
N_start_point                \                    /
                              \                  /
        Mask (M):              \                /
            3 4 (5) 4 3 -------(*) = 3+8+(15)+16+15 = weight sum


        - missing elements are referred to as ghost elements



Cache coherence mechanism
    -  needed to ensure that the contents of the caches of the other processor cores are updated

*/

#define MASK_WIDTH 5

int main()
{
    float h_M[] = [ 1.0, 2.0, 3.0, 4.0, 5.1 ];

    // declare constant mem in host
    __constant__ float M[MASK_WIDTH];

    // transferred the contents of the h_M to device constant mem M
    // constant memory will not be changed during kernel execution
    // cudaMemcpyToSymbol(dest, src, size)
    cudaMemcpyToSymbol(M, h_M, MASK_WIDTH * sizeof(float));

    ConvolutionKernel<<<dimGrid, dimBlock>>>(Nd, Pd);
}

__global__ void convolution_1D_basic_kernel(float *N, float *P, int Mask_Width, int Width)
{
    // output element index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue = 0;

    //              0 = 2 - 5 /2
    int N_start_point = i - (Mask_Width / 2);

    for (int j = 0; j < Mask_Width; j++)
    {
        if (N_start_point + j >= 0 && N_start_point + j < Width)
        {
            Pvalue += N[N_start_point + j] * M[j]; // <-- constant mem M is a global var, can access here
        }
    }

    P[i] = Pvalue;
}

__global__ void convolution_1D_tiled_kernel(float *N, float *P, int Mask_Width, int Width)
{
    // for P[6]
    // Tile 1:                               ['2' '3' 4 5 6 7 '8' '9']  will be stored in N_ds
    // block size = 4
    // i  = 1          x 4          + 2 = 6
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //                      4 + 5 - 1 = 8
    __shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];

    int n = Mask_Width / 2; // = 5/2 = 2

    //                2 =        (1 - 1) x 4 + 2
    int halo_index_left = (blockIdx.x - 1) * blockDim.x + threadIdx.x;

    // 2 >= (4 - 2)
    if (threadIdx.x >= blockDim.x - n)
    {
        // thread p[6] loads left halo element '2'
        //      2 - (4 - 2) = 0                         2 < 0 ? 0 : N[2]
        // N_ds[0] = N[2]
        N_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left < 0) ? 0 : N[halo_index_left];
    }

    //   2 + 2                  1        x   4        + 2
    // N_ds[4] = N[6]
    N_ds[n + threadIdx.x] = N[blockIdx.x * blockDim.x + threadIdx.x];

    int halo_index_right = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

    if (threadIdx.x < n)
    {
        N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right >= Width) ? 0 : N[halo_index_right];
    }

    __syncthreads();

    float Pvalue = 0;

    for (intj = 0; j < Mask_Width; j++)
    {
        Pvalue += N_ds[threadIdx.x + j] * M[j]; // <-- constant mem M is a global var, can access here
    }

    P[i] = Pvalue;
}

__global__ void convolution_1D_tiled_general_cache_kernel(float *N, float *P, int Mask_Width, int Width)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float N_ds[TILE_SIZE];

    N_ds[threadIdx.x] = N[i];

    __syncthreads();

    int This_tile_start_point = blockIdx.x * blockDim.x;
    int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
    int N_start_point = i - (Mask_Width / 2);

    float Pvalue = 0;
    for (int j = 0; j < Mask_Width; j++)
    {
        int N_index = N_start_point + j;
        if (N_index >= 0 && N_index < Width)
        {
            if ((N_index >= This_tile_start_point) && (N_index < Next_tile_start_point))
            {
                Pvalue += N_ds[threadIdx.x + j - (Mask_Width / 2)] * M[j];
            }
            else
            {
                // N[N_index] in general caching (L2 cache), M in constant mem
                Pvalue += N[N_index] * M[j]; // <-- constant mem M is a global var, can access here
            }
        }
    }

    P[i] = Pvalue;
}

/* -------------------------------- from slides -----------------------------------------*/

#define MASK_WIDTH 5

// Matrix Structure declaration
typedef struct
{
    unsigned int width;
    unsigned int height;
    unsigned int pitch;
    float *elements;
} Matrix;

Matrix AllocateMatrix(int height, int width, int init)
{
    Matrix M;

    M.width = M.pitch = width;
    M.height = height;
    int size = M.width * M.height;
    M.elements = NULL;

    // don't allocate memory on option 2
    if (init == 2)
        return M;

    int size = height * width;
    M.elements = (float *)malloc(size * sizeof(float));

    for (unsigned int i = 0; i < M.height * M.width; i++)
    {
        M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
        if (rand() % 2)
            M.elements[i] = -M.elements[i]
    }

    return M;
}