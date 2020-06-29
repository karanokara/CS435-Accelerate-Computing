#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <driver_types.h>


// Compute vector multiplication C = A x B
// Each thread performs one pair-wise addition
__global__ void device_multiply(long *a,long *b, long *c, long N) {

    long data_value = blockIdx.x *  blockDim.x + threadIdx.x;

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
int allocate_device_mem(long **dev_arr, long N) {
    cudaError_t err = cudaMalloc((void**)dev_arr, N * sizeof(long));
    
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return 0;
    }

    return 1;
}

// add up product and print the sum
long get_add_up_product(long *arr, long N) {
    long sum = 0;
	for(long i=0; i<N; ++i) {
        sum += arr[i];
    }

    return sum;
}

// print out the capability of current device being used
void print_device_capability() {
    // Get grid (device) info
    int device;
    struct cudaDeviceProp prop;
    cudaGetDevice(&device);                 // get current working device index
    cudaGetDeviceProperties(&prop, device);

    printf("Current device compute capability: major(%d) minor(%d).\n", prop.major, prop.minor);
}

int perform_on_device(long *a, long *b, long *c, long N, long B, long T, float *elapsed_time_ms ) {
	long *dev_a, *dev_b, *dev_c; // allocate in device Mem

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


	cudaMemcpy(dev_a, a , N*sizeof(long),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b , N*sizeof(long),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_c, c , N*sizeof(long),cudaMemcpyHostToDevice);

    // instrument code to measure start time
    cudaEventCreate( &start );     
	cudaEventCreate( &stop );
	cudaEventRecord( start, 0 );

    // 2
    // Kernel launch code – the device performs the actual vector addition
    device_multiply<<<B,T>>>(dev_a,dev_b,dev_c, N);

    // 3
    // copy C from the device memory
    cudaMemcpy(c,dev_c,N*sizeof(long),cudaMemcpyDeviceToHost);
    
    // instrument code to measue end time
	cudaEventRecord( stop, 0 );     
	cudaEventSynchronize( stop );
	cudaEventElapsedTime( elapsed_time_ms, start, stop );

	// Free device vectors
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 1;
}

void perform_on_host(long *a, long *b, long *c, long N, float *elapsed_time_ms ) {
    // cudaEvent_t start, stop;       // using cuda events to measure time
    //                                // which is applicable for asynchronous code also
                                   
    // // instrument code to measure start time
    // cudaEventCreate( &start );
	// cudaEventCreate( &stop );
    // cudaEventRecord( start, 0 );

    struct timeval start, end;
    double elapsedTime;
    gettimeofday(&start, 0);


    for(long i = 0; i < N; ++i){
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

int main(int argc, char *argv[])  {
    long T = 10;
    long B = 1;      // threads per block and blocks per grid
    long N = 0;      // vector size
	float host_elapsed_time_ms;
	float device_elapsed_time_ms;

    
	printf("Size of array = %ld\n", N);
	do {
        printf("Enter number of vector size: ");
        scanf("%ld",&N);
        
		printf("Enter number of threads per block: ");
		scanf("%ld",&T);
        
        printf("\nEnter number of blocks per grid: ");
		scanf("%ld",&B);
        
        if (T * B != N)
            printf("Error T x B != N, try again");
        
    } while (T * B != N);
    
    printf("--------------\n");

    long a[N], b[N], host_c[N], device_c[N];
    printf("--------------\n");

    for(long i=0; i<N; ++i) {    // load arrays with some numbers
		a[i] = i;
		b[i] = i;
	}
    printf("--------------\n");

    print_device_capability();

    perform_on_host(a, b, host_c, N, &host_elapsed_time_ms);

    if(perform_on_device(a, b, device_c, N, B, T, &device_elapsed_time_ms) == 0)
        exit(EXIT_FAILURE);


    // add up product, host
    long host_sum = get_add_up_product(host_c, N);

    // add up product, device
    long device_sum = get_add_up_product(device_c, N);

    // print out execution time
    printf("Using CPU:\n   sum of product: %ld\n   time to calculate product: %f ms.\n", host_sum, host_elapsed_time_ms);
    printf("Using GPU:\n   sum of product: %ld\n   time to calculate product: %f ms.\n", device_sum, device_elapsed_time_ms);
	
	return 0;
}

