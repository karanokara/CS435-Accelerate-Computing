#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h>  // for printf
#include <time.h>   // for nanosleep
#include <sys/time.h>

int saxpy(int N, float a, float *y, float *x)
{
    int b;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
        {
            b = a * x[i] + y[j]; // this loop has NO Data dependency
        }

    return b;
}

int saxpy_kernels(int N, float a, float *restrict y, float *restrict x)
{
    int b;

// Loop is parallelizable
#pragma acc kernels
    {
        // #pragma acc loop gang, vector(4) /* blockIdx.x threadIdx.x */
        for (int j = 0; j < N; ++j)
            // #pragma acc loop gang, vector(32) /* blockIdx.x threadIdx.x */
            for (int i = 0; i < N; ++i)
            {
                b = a * x[i] + y[j]; // this loop has NO Data dependency
            }
    }
}

int saxpy_parallel_loop(int N, float a, float *restrict y, float *restrict x)
{
    int b;

// Loop is parallelizable
#pragma acc parallel loop
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i)
        {
            b = a * x[i] + y[j]; // this loop has NO Data dependency
        }
}

// same compilation as above
int saxpy_parallel_loop_loop(int N, float a, float *restrict y, float *restrict x)
{
    int b;

// Loop is parallelizable
#pragma acc parallel loop
    for (int j = 0; j < N; ++j)
#pragma acc loop
        for (int i = 0; i < N; ++i)
        {
            b = a * x[i] + y[j]; // this loop has NO Data dependency
        }
}

// declare start, end
// struct timeval start, end;
// start counting
void count_start(struct timeval *start)
{
    gettimeofday(start, 0);
}

// end counting
// return elapsed time in ms
double count_end(struct timeval *start, struct timeval *end)
{
    double elapsedTime;

    gettimeofday(end, 0);

    elapsedTime = (end->tv_sec - start->tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (end->tv_usec - start->tv_usec) / 1000.0; // us to ms

    return elapsedTime;
}

int main(int argc, char **argv)
{
    int N = 100000;
    float a = 3.0f;
    float x[N], y[N];

    // float *x_ptr = (float *)malloc(N * sizeof(float));
    // float *y_ptr = (float *)malloc(N * sizeof(float));

    float *restrict x_restrict_ptr = (float *)malloc(N * sizeof(float));
    float *restrict y_restrict_ptr = (float *)malloc(N * sizeof(float)); //  avoid pointer aliasing

    for (int i = 0; i < N; ++i)
    {
        x[i] = 2.0f;
        y[i] = 1.0f;
        x_restrict_ptr[i] = 2.0f;
        y_restrict_ptr[i] = 1.0f;
    }

    struct timeval start, end;

    count_start(&start);
    saxpy(N, a, y_restrict_ptr, x_restrict_ptr);
    printf("Time elapsed saxpy: %f ms.\n", count_end(&start, &end));

    count_start(&start);
    saxpy_kernels(N, a, y_restrict_ptr, x_restrict_ptr);
    printf("Time elapsed saxpy_kernels: %f ms.\n", count_end(&start, &end));

    count_start(&start);
    saxpy_parallel_loop(N, a, y_restrict_ptr, x_restrict_ptr);
    printf("Time elapsed saxpy_parallel_loop: %f ms.\n", count_end(&start, &end));

    count_start(&start);
    saxpy_parallel_loop_loop(N, a, y_restrict_ptr, x_restrict_ptr);
    printf("Time elapsed saxpy_parallel_loop_loop: %f ms.\n", count_end(&start, &end));

    free(x);
    free(y);
    free(x_restrict_ptr);
    free(y_restrict_ptr);
}