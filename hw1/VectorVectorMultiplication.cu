#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <driver_types.h>

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

// Compute vector multiplication C = A x B
// Each thread performs one pair-wise addition
__global__ void device_multiply(int *a,int *b, int *c, int N) {

    int data_value = blockIdx.x *  blockDim.x + threadIdx.x;

    // data_value index will be multiple of 32, there are extra threads
    if(data_value < N){
        c[data_value] = a[data_value] * b[data_value];
    }
}

// perform multiplication using CPU
// void host_multiply(long *a, long *b, long *c, long N) {

//     long i;
//     for( i = 0; i < N; ++i){
//         c[i] = a[i] * b[i];
//     }
// }

// traditional host function (C style)
int allocate_device_mem(int **dev_arr, int N) {
    cudaError_t err = cudaMalloc((void**)dev_arr, N * sizeof(int));
    
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return 0;
    }

    return 1;
}

// add up product and print the sum
int get_add_up_product(int *arr, int N) {
    int sum = 0;
	for(int i=0; i<N; ++i) {
        sum += arr[i];
    }

    return sum;
}

// print out the capability of current device being used
int print_device_capability() {
    // Get grid (device) info
    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice(&device);                 // get current working device index
    cudaGetDeviceProperties(&prop, device);

    printf("Current device compute capability: major(%d) minor(%d).\n", prop.major, prop.minor);
    printf("Device Max # thread per block: %d\n", prop.maxThreadsPerBlock);

    return prop.maxThreadsPerBlock;
}

int perform_on_device(int *a, int *b, int *c, int N, int B, int T, float *elapsed_time_ms ) {
	int *dev_a, *dev_b, *dev_c; // allocate in device Mem

    cudaEvent_t start, stop;       // using cuda events to measure time
	                               // which is applicable for asynchronous code also

    // 1
    // Allocate device memory for A, B, and C
	if(allocate_device_mem(&dev_a,N) == 0)
        return 0;
    
    if(allocate_device_mem(&dev_b,N) == 0)
        return 0;

    if(allocate_device_mem(&dev_c,N) == 0)
        return 0;


	cudaMemcpy(dev_a, a , N*sizeof(int),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b , N*sizeof(int),cudaMemcpyHostToDevice);
	// cudaMemcpy(dev_c, c , N*sizeof(int),cudaMemcpyHostToDevice);

    // measure start time
    cuda_measure_start(&start, &stop);

    // 2
    // Kernel launch code – the device performs the actual vector addition
    device_multiply<<<B,T>>>(dev_a,dev_b,dev_c, N);

    // 3
    // copy C from the device memory
    cudaMemcpy(c,dev_c,N*sizeof(int),cudaMemcpyDeviceToHost);
    
    // measure end time
    cuda_measure_stop(&start, &stop, elapsed_time_ms);

	// Free device vectors
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	// cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    
    return 1;
}

void perform_on_host(int *a, int *b, int *c, int N, float *elapsed_time_ms ) {
    // cudaEvent_t start, stop;       // using cuda events to measure time
    //                                // which is applicable for asynchronous code also
                                   
    // // instrument code to measure start time
    // cudaEventCreate( &start );
	// cudaEventCreate( &stop );
    // cudaEventRecord( start, 0 );

    struct timeval start, end;
    double elapsedTime;
    gettimeofday(&start, 0);


    for(int i = 0; i < N; ++i){
        c[i] = a[i] * b[i];
    }

    gettimeofday(&end, 0);
    // timerinterval(&start, &end, &interval);

    elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;   // us to ms

    // *elapsed_time_ms = (((float)interval.tv_sec) + ((float)interval.tv_usec) / 1000000)) * 1000;
    *elapsed_time_ms = (float)elapsedTime;

    // // instrument code to measue end time
	// cudaEventRecord( stop, 0 );     
	// cudaEventSynchronize( stop );
	// cudaEventElapsedTime( elapsed_time_ms, start, stop );
}

/*
Run:
./VectorVectorMultiplication

vector size: 100000
block: 128
blocks per grid: 1000

Using CPU:
   sum of product: 216474736
   time to calculate product: 0.646000 ms.
Using GPU:
   sum of product: 216474736
   time to calculate product: 0.243936 ms.

*/
int main(int argc, char *argv[])  {
    int T = 0;
    int B = 0;      // threads per block and blocks per grid
    int N = 0;      // vector size
	float host_elapsed_time_ms;
	float device_elapsed_time_ms;
    int max_thread = print_device_capability();

	do {
        printf("Enter number of vector size: ");
        scanf("%d",&N);
        
		printf("Enter number of threads per block: ");
		scanf("%d",&T);
        
        printf("\nEnter number of blocks per grid: ");
        scanf("%d",&B);
        
        if (T > max_thread)
            printf("Error T > %d, try again\n", max_thread);
        
        if (T * B < N)
            printf("Error T x B < N, try again\n");
        
    } while ((T * B < N) || (T > max_thread));
    
    printf("Size of array = %d\n", N);

    int a[N], b[N], host_c[N], device_c[N];

    for(int i=0; i<N; ++i) {    // load arrays with some numbers
		a[i] = i;
		b[i] = i;
	}


    perform_on_host(a, b, host_c, N, &host_elapsed_time_ms);

    if(perform_on_device(a, b, device_c, N, B, T, &device_elapsed_time_ms) == 0)
        exit(EXIT_FAILURE);


    // add up product, host
    int host_sum = get_add_up_product(host_c, N);

    // add up product, device
    int device_sum = get_add_up_product(device_c, N);

    // print out execution time
    printf("Using CPU:\n   sum of product: %d\n   time to calculate product: %f ms.\n", host_sum, host_elapsed_time_ms);
    printf("Using GPU:\n   sum of product: %d\n   time to calculate product: %f ms.\n", device_sum, device_elapsed_time_ms);
	
	return 0;
}

