#include <stdio.h>
#include <cuda.h>       // need cuda library
#include <stdlib.h>
#include <time.h>
#include <driver_types.h>

#define N 4096         // size of array

void measure_start(cudaEvent_t *start, cudaEvent_t *stop)
{
    cudaEventCreate(start);
    cudaEventCreate(stop);
    cudaEventRecord(*start, 0);
}

void measure_stop(cudaEvent_t *start, cudaEvent_t *stop, float *elapsed_time_ms)
{
    cudaEventRecord(*stop, 0); // instrument code to measue end time
    cudaEventSynchronize(*stop);
    cudaEventElapsedTime(elapsed_time_ms, *start, *stop);

    cudaEventDestroy(*start);
    cudaEventDestroy(*stop);
}

// Compute vector sum C = A+B
// Each thread performs one pair-wise addition
__global__ void add(int *a,int *b, int *c) {
    // gridDim.x    # of block in X coor in a grid   == "B" in main() 
    // blockDim.x - # of thread in X corr in a block == "T" in main()
    // blockIdx.x - the block id in a grid, range in 0 ~ gridDim.x-1
    // threadIdx.x - thread id in a block
    
    // Each thread uses data_value index to access d_A, d_B, and d_C
    // Different threads will see different values of threadIdx.x, blockIdx.x, blockDim.x
    // one thread will be identified as "thread (blockIdx.x, threadIdx.x)"
    // data_value index is an automatic variable that is private to each thread
    int data_value = blockIdx.x *  blockDim.x + threadIdx.x;
    

    // data_value index will be multiple of 32, there are extra threads
    if(data_value < N){
        c[data_value] = a[data_value]+b[data_value];
    }
}

int allocate_device_mem(int **dev) {
    //                                               4       
    cudaError_t err = cudaMalloc((void**)dev, N * sizeof(int));
    
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        // exit(EXIT_FAILURE);
        return 0;
    }

    return 1;
}

// print out the capability number of the device being used
void print_device_capability() {
    int deviceCount;

    // get # of available CUDA devices (# of GPU) in system
    cudaGetDeviceCount(&deviceCount);   

    printf("\nDevices compute capability:\n");

    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device(%d) has compute capability: major(%d) minor(%d).\n", device, deviceProp.major, deviceProp.minor);
        printf("# of thread per Block in this device: %d \n", deviceProp.maxThreadsPerBlock);
        printf("# of SMs in this device: %d \n", deviceProp.multiProcessorCount);
        printf("# of blcok per SMs: Unknown \n");
        printf("# of shared Mem per Block: %ld bytes\n", deviceProp.sharedMemPerBlock);
        printf("# of constant Mem per Grid: %ld bytes\n", deviceProp.totalConstMem);
        printf("Clock Frequency of this device: %d \n", deviceProp.clockRate);
        printf("# of threads in dimension: %d %d %d \n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Warp size of this device: %d threads/warp \n", deviceProp.warpSize);

    }
}

int main(int argc, char *argv[])  {
	int T = 10, B = 1;            // threads per block and blocks per grid
	int a[N],b[N],c[N];
    int *dev_a, *dev_b, *dev_c;
    
    print_device_capability();

	printf("Size of array = %d\n", N);
	do {
		printf("Enter number of threads per block: ");
		scanf("%d",&T);
		printf("\nEnter number of blocks per grid: ");
		scanf("%d",&B);
		if (T * B != N) printf("Error T x B != N, try again");
	} while (T * B != N);

	cudaEvent_t start, stop;     // using cuda events to measure time
	float elapsed_time_ms;       // which is applicable for asynchronous code also

    // 1
    // Allocate device memory for A, B, and C
	// cudaError_t err1 = cudaMalloc((void**)&dev_a, N * sizeof(int));
	// cudaError_t err2 = cudaMalloc((void**)&dev_b, N * sizeof(int));
    // cudaError_t err3 = cudaMalloc((void**)&dev_c, N * sizeof(int));
    
    // if to check error:
    if(allocate_device_mem(&dev_a) == 0)
        exit(EXIT_FAILURE);
    
    if(allocate_device_mem(&dev_b) == 0)
        exit(EXIT_FAILURE);

    if(allocate_device_mem(&dev_c) == 0)
        exit(EXIT_FAILURE);


	for(int i=0;i<N;i++) {    // load arrays with some numbers
		a[i] = i;
		b[i] = i*1;
	}

    // 2 
    // copy data to device memory
    // cudamemcpy(p_dest, p_source, #byte, Direction)
	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c, N*sizeof(int), cudaMemcpyHostToDevice);

	// cudaEventCreate( &start );     // instrument code to measure start time
	// cudaEventCreate( &stop );
    // cudaEventRecord( start, 0 );
    
    // measure start time
    measure_start(&start, &stop);

    // 3
    // Kernel launch code – the device performs the actual vector addition
    // here host code launches a kernel, it sets the config param
    // Execution configuration parameters: <<< # of thread block, # of thread each block >>>
    add<<< B, T >>>(dev_a, dev_b, dev_c);
    // #block = ceil("threads needed" / "threads per block GPU has")
    // vecAddKernel<<< ceil(n/256.0), 256 >>>(d_A, d_B, d_C, n);


    // 4
    // copy data from the device memory to host memory
	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

	// cudaEventRecord( stop, 0 );     // instrument code to measue end time
	// cudaEventSynchronize( stop );
    // cudaEventElapsedTime( &elapsed_time_ms, start, stop );
    
    // measure end time
    measure_stop(&start, &stop, &elapsed_time_ms);

    // print out each addition result
	for(int i=0;i<N;i++) {
		printf("%d+%d=%d\n",a[i],b[i],c[i]);
	}

    // Get grid (device) info
    // int device;
    // struct cudaDeviceProp props;
    // cudaGetDevice(&device);                 // get current working device index
    // cudaGetDeviceProperties(&props, device);


	printf("Time to calculate results: %f ms.\n", elapsed_time_ms);  // print out execution time


    // Free device vectors
    // cudaFree(pointer to freed obj)
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	return 0;
}

