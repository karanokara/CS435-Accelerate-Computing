#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <driver_types.h>

#define N 4096         // size of array

// Compute vector sum C = A+B
// Each thread performs one pair-wise addition
__global__ void add(int *a,int *b, int *c) {
    // blockDim.x - # of thread in a block
    // threadIdx.x - thread id in a block
    // blockIdx.x - the block id in a grid
    
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
    cudaGetDeviceCount(&deviceCount);

    printf("\nDevices compute capability:\n");

    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device(%d) has compute capability: major(%d) minor(%d).\n", device, deviceProp.major, deviceProp.minor);
        printf("Device Max # thread per block: %d\n", deviceProp.maxThreadsPerBlock);
    }
}

int main(int argc, char *argv[])  {
	int T = 10, B = 1;            // threads per block and blocks per grid
	int a[N],b[N],c[N];
	int *dev_a, *dev_b, *dev_c;

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

	cudaMemcpy(dev_a, a , N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b , N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c , N*sizeof(int),cudaMemcpyHostToDevice);

	cudaEventCreate( &start );     // instrument code to measure start time
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

    // 2
    // Kernel launch code – the device performs the actual vector addition
    // here host code launches a kernel, it sets the config param
    // Execution configuration parameters: <<< # of thread block, # of thread each block >>>
    add<<<B,T>>>(dev_a,dev_b,dev_c);
    // #block = ceil("threads needed" / "threads per block GPU has")
    // vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);


    // 3
    // copy C from the device memory
	cudaMemcpy(c,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost);

	cudaEventRecord( stop, 0 );     // instrument code to measue end time
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( &elapsed_time_ms, start, stop );

    // print out each addition result
	for(int i=0;i<N;i++) {
		printf("%d+%d=%d\n",a[i],b[i],c[i]);
	}

    // Get grid (device) info
    int device;
    struct cudaDeviceProp props;
    cudaGetDevice(&device);                 // get current working device index
    cudaGetDeviceProperties(&props, device);

    // print out device properties
    printf("Device(%d) Property: major(%d) minor(%d) \n", device, props.major, props.minor ); 


	printf("Time to calculate results: %f ms.\n", elapsed_time_ms);  // print out execution time


	// Free device vectors
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}

