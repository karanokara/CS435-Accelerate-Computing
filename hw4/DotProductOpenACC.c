#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

/*
1. Write an OpenACC code to compute the dot product of two vectors. 
The dot product is the sum of the products of each pair of elements. 
The result is a single value.
 */

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

// perform dot product of two vectors
int calculate(int *a, int *b, int N)
{
    int sum = 0;

    struct timeval start, end;
    count_start(&start);

    for (int i = 0; i < N; ++i)
    {
        sum += a[i] * b[i];
    }

    printf("Sum: %d    Time elapsed using CPU: %f ms.\n", sum, count_end(&start, &end));

    return sum;
}

// perform dot product of two vectors using openacc
int calculate_openacc_parallel(int *a, int *b, int N)
{
    int sum = 0;

    struct timeval start, end;
    count_start(&start);

#pragma acc parallel loop copyin(a[:N], b[:N]) reduction(+ \
                                                         : sum)
    for (int i = 0; i < N; ++i)
    {
        sum = a[i] * b[i];
    }

    printf("Sum: %d    Time elapsed using OpenACC parallel loop: %f ms.\n", sum, count_end(&start, &end));

    return sum;
}

// perform dot product of two vectors using openacc
int calculate_openacc_kernel(int *a, int *b, int N)
{
    int sum = 0;

    struct timeval start, end;
    count_start(&start);

#pragma acc kernels
    for (int i = 0; i < N; ++i)
    {
        sum += a[i] * b[i];
    }

    printf("Sum: %d    Time elapsed using OpenACC kernels: %f ms.\n", sum, count_end(&start, &end));

    return sum;
}

int main(int argc, char *argv[])
{
    int N = 0; // vector size

    do
    {
        printf("Enter number of vector size: ");
        scanf("%d", &N);

    } while (N <= 0);

    printf("Size of array = %d\n", N);

    int a[N], b[N];

    for (int i = 0; i < N; ++i)
    {
        // load arrays with some numbers
        a[i] = i;
        b[i] = i;
    }

    calculate(a, b, N);
    // calculate_openacc_kernel(a, b, N);
    calculate_openacc_parallel(a, b, N);

    return 0;
}
