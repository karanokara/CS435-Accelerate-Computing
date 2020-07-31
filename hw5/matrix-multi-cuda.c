/*
Perform matrix multiplication on 2 2-D matrices using CUDA non-tiled
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <driver_types.h>
#include <time.h>
#include <sys/time.h>

#define WIDTH 1000

__global__ void matrix_mult_kernel(int *m, int *n, int *p, int width);
void device_prop_print(int print);
void printDevProp(cudaDeviceProp devProp);
void cuda_measure_start(cudaEvent_t *start, cudaEvent_t *stop);
void cuda_measure_stop(cudaEvent_t *start, cudaEvent_t *stop, float *elapsed_time_ms);

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        printf("Usage %s vector_width gird_width block_width\n", argv[0]);
        exit(1);
    }

    int display_matrix = 0;
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;
    int vector_width = atoi(argv[1]);
    int grid_width = atoi(argv[2]);
    int block_width = atoi(argv[3]);

    int N = vector_width * vector_width;
    int size = N * sizeof(int);

    cudaEvent_t start, stop; // using cuda events to measure time
    float elapsed_time_ms;

    if ((block_width * block_width * grid_width * grid_width) < (vector_width * vector_width))
    {
        printf("Error block_width^2 x grid_width^2 < vector_width^2, try again!\n");
        return 1;
    }

    device_prop_print(0);

    printf("Perform Matrix Multiplication on [%d x %d] x [%d x %d]\n", vector_width, vector_width, vector_width, vector_width);

    h_A = (int *)malloc(sizeof(int) * N);
    h_B = (int *)malloc(sizeof(int) * N);
    h_C = (int *)malloc(sizeof(int) * N);

    for (int i = 0; i < N; ++i)
    {
        h_A[i] = 1;
        h_B[i] = 1;
        h_C[i] = 2;
    }

    if (display_matrix)
    {
        for (int i = 0; i < N; ++i)
        {
            if (i > 0 && i % WIDTH == 0)
                printf("\n");
            printf("%i ", h_A[i]);
        }

        printf("\n");
        for (int i = 0; i < N; ++i)
        {
            if (i > 0 && i % WIDTH == 0)
                printf("\n");
            printf("%i ", h_B[i]);
        }
        printf("\n");
    }

    cuda_measure_start(&start, &stop);
    // ------------------------- start timing --------------------------- //

    // Allocate device memory and copy to device
    cudaMalloc((void **)&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&d_C, size);

    // Grid and block dimensions
    // dim3 dim_grid(ceil(N/16), ceil(N/16), 1);
    // dim3 dim_block(16, 16, 1);
    dim3 dim_grid(grid_width, grid_width, 1);
    dim3 dim_block(block_width, block_width, 1);

    // CUDA kernel execution
    // cudaEventRecord(start, 0);
    matrix_mult_kernel<<<dim_grid, dim_block>>>(d_A, d_B, d_C, vector_width);

    // Copy 2d array from device back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // ------------------------- end timing --------------------------- //
    cuda_measure_stop(&start, &stop, &elapsed_time_ms);

    if (display_matrix)
    {
        for (int i = 0; i < N; ++i)
        {
            if (i > 0 && i % WIDTH == 0)
                printf("\n");
            printf("%i ", h_C[i]);
        }
    }

    printf("Using GPU:\n    time to calculate: %f ms.\n", elapsed_time_ms);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

__global__ void matrix_mult_kernel(int *m, int *n, int *p, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((row < width) && (col < width))
    {
        int p_value = 0;

        for (int i = 0; i < width; ++i)
        {
            p_value += m[row * width + i] * n[i * width + col];
        }
        p[row * width + col] = p_value;
    }
}

void device_prop_print(int print)
{
    if (print)
    {
        int devCount;
        cudaGetDeviceCount(&devCount);
        printf("CUDA Device Query...\n");
        printf("There are %d CUDA devices.\n", devCount);

        for (int i = 0; i < devCount; ++i)
        {
            // Get device properties
            printf("\nCUDA Device #%d\n", i);
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, i);
            printDevProp(devProp);
        }
    }
}

void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n", devProp.major);
    printf("Minor revision number:         %d\n", devProp.minor);
    printf("Name:                          %s\n", devProp.name);
    printf("Total global memory:           %lu\n", devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n", devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n", devProp.regsPerBlock);
    printf("Warp size:                     %d\n", devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n", devProp.memPitch);
    printf("Maximum threads per block:     %d\n", devProp.maxThreadsPerBlock);

    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);

    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);

    printf("Clock rate:                    %d\n", devProp.clockRate);
    printf("Total constant memory:         %lu\n", devProp.totalConstMem);
    printf("Texture alignment:             %lu\n", devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n", devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    return;
}

void cuda_measure_start(cudaEvent_t *start, cudaEvent_t *stop)
{
    cudaEventCreate(start);
    cudaEventCreate(stop);
    cudaEventRecord(*start, 0);
}

void cuda_measure_stop(cudaEvent_t *start, cudaEvent_t *stop, float *elapsed_time_ms)
{
    cudaEventRecord(*stop, 0); // instrument code to measue end time
    cudaEventSynchronize(*stop);
    cudaEventElapsedTime(elapsed_time_ms, *start, *stop);

    cudaEventDestroy(*start);
    cudaEventDestroy(*stop);
}