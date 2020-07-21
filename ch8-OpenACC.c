/*

OpenACC (open accelerators)

- Directive-based acceleration
    - little code modification
    - source code can be compiled with/without acceleration


- Usage (in source code)
    
    #pragma acc directive [clause [,] clause] ...]...


    - 3 kinds Driectives
        1. compute region
            1. kernels
                - #pragma acc kernels
                - let compiler to Generate parallel code for GPU, accelerate IF possible
                - including copy in/out data from CPU to GPU

            2. parallel
                - #pragma acc parallel
                - tell compiler to parallelize the code

            3. loop
                - #pragma acc loop
                - Tells compiler there are no dependencies in this loop, it can be parallelized

            4. routine
                - #pragma acc routine
                - Calls function from parallel regions, compile to use the device

        2. data region
            - copy (array): allocate, copyin, copyout data between host and GPU 

            - copyin (array): allocate, copyin data from host to GPU when enter

            - copyout (array): allocate, copyout data from GPU to host when exit

            - create (array): allocate mem on GPU

            - present (array): array is already present on GPU


        3. synchronization

    - Clauses:
        - Data handling
        - work distribution across threads
        - control flow




- To compile

pgcc -acc -Minfo=accel hello.c -o hello

    -acc                tell compiler not to ignore #pragma
    -Minfo=accel        show compiling info

export PGI_ACC_TIME=1
    - to activate profiling

    Terms in OpenACC        in CUDA
    gang                    block
    worker                  warp
    vector                  thread



*/

// OpenACC runtime API
// do what???
#include "openacc.h"

int main(int argc, char **argv)
{
    int N = 1000;
    float a = 3.0f;
    float x[N], y[N];

    float *x_ptr = (float *)malloc(N * sizeof(float));
    float *y_ptr = (float *)malloc(N * sizeof(float));

    float *x_ptr = (float *)malloc(N * sizeof(float));
    float *restrict y_restrict_ptr = (float *)malloc(N * sizeof(float)); //  avoid pointer aliasing

    for (int i = 0; i < N; ++i)
    {
        x[i] = 2.0f;
        y[i] = 1.0f;
    }

// Loop is parallelizable
// #pragma acc kernels
#pragma acc loop gang, vector(128) // <-- compiler change to this
    {
        for (int i = 0; i < N; ++i)
        {
            y[i] = a * x[i] + y[i]; // this loop has NO Data dependency
        }
    }

// #pragma acc kernels
#pragma acc parallel loop //
    {
        for (int i = 0; i < N; ++i)
        {
            y[i] = a * x[i] + y[i]; // this loop has NO Data dependency
        }
    }

// Loop is Not parallelizable
#pragma acc kernels
    {
        for (int i = 0; i < N; ++i)
        {
            x[i] = a * x[i + 1]; // this loop has Data dependency, i depend on i+1
                                 // need to compute serially
        }
    }

// Loop is Not parallelizable
#pragma acc kernels
    {
        for (int i = 0; i < N; ++i)
        {
            // if x_ptr = y_ptr, then has Data dependency
            y_ptr[i] = a * x_ptr[i] + y_ptr[i]; // Pointer aliasing
        }
    }

// Loop is parallelizable
#pragma acc kernels
    {
        for (int i = 0; i < N; ++i)
        {
            // y_restrict_ptr and x_ptr points to different obj
            y_restrict_ptr[i] = a * x_ptr[i] + y_restrict_ptr[i];
        }
    }
}

int two_kernels()
{
    int n = 1000;
    int a[n], b[n];

// Generate two kernels
#pragma acc kernels
    {
        for (int i = 0; i < n; i++)
            a[i] = 3.0f * (float)(i + 1);

        for (int i = 0; i < n; i++)
            b[i] = 2.0f * a[i];
    }

// Generate one kernel, but inner loop may not run in order, incorrected
#pragma acc parallel
    {
#pragma acc loop
        for (int i = 0; i < n; i++)
            a[i] = 3.0f * (float)(i + 1);

#pragma acc loop
        for (int i = 0; i < n; i++)
            b[i] = 2.0f * a[i];
    }
}

// Serial code: 17.610445 seconds
// this:        48.796347 seconds
//
// 4 times data transfer between host(CPU) memory and GPU memory in
//every iteration of the outer while loop.
int Laplace_Solver_v1(int iteration)
{
    int n = 1000;
    int MAX_TEMP_ERROR = 1000;
    int dt;
    int max_iterations;
    int A_new[n][n], A[n][n];
    int ROWS[n], COLUMNS[n];

    while (dt > MAX_TEMP_ERROR && iteration <= max_iterations)
    {
        dt = 0.0;

#pragma acc kernels
        // data copyin from host to GPU
        //  computing a new solution
        for (int i = 1; i <= ROWS; i++)
            for (int j = 1; j <= COLUMNS; j++)
            {
                A_new[i][j] = 0.25 * (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]);
            }
            // data copyout from GPU to host

#pragma acc kernels
        // data copyin from host to GPU
        //  updating the solution and finding the max error.
        for (int i = 1; i <= ROWS; i++)
            for (int j = 1; j <= COLUMNS; j++)
            {
                dt = fmax(fabs(A_new[i][j] - A[i][j]), dt);
                A[i][j] = A_new[i][j];
            }
        // data copyout from GPU to host

        iteration++;
    }
}

// Serial code: 17.610445 seconds
// this:        2.592581 seconds
int Laplace_Solver_v2(int iteration)
{
    int n = 1000;
    int MAX_TEMP_ERROR = 1000;
    int dt;
    int max_iterations;
    int A_new[n][n], A[n][n];
    int ROWS[10], COLUMNS[10];

// copyin/out array of A for the outer loop, create array A_new on GPU
// Create a data region
#pragma acc data copy(A), create(A_new)
    while (dt > MAX_TEMP_ERROR && iteration <= max_iterations)
    {

#pragma acc kernels // Create a kernel region
        //  computing a new solution
        for (int i = 1; i <= ROWS; i++)
            for (int j = 1; j <= COLUMNS; j++)
            {
                A_new[i][j] = 0.25 * (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]);
            }

        dt = 0.0;

#pragma acc kernels // Create a kernel region
        //  updating the solution and finding the max error.
        for (int i = 1; i <= ROWS; i++)
            for (int j = 1; j <= COLUMNS; j++)
            {
                dt = fmax(fabs(A_new[i][j] - A[i][j]), dt);
                A[i][j] = A_new[i][j];
            }

        iteration++;
    }
}

// Serial code: 17.610445 seconds
// this:        2.259797 seconds
// fastest
int Laplace_Solver_v3(int iteration)
{
    int n = 1000;
    int MAX_TEMP_ERROR = 1000;
    int dt;
    int max_iterations;
    int A_new[n][n], A[n][n];
    int ROWS[10], COLUMNS[10];

// copyin/out array of A for the outer loop, create array A_new on GPU
// Create a data region
#pragma acc data copy(A), create(A_new)
    while (dt > MAX_TEMP_ERROR && iteration <= max_iterations)
    {

#pragma acc parallel loop // Create a "parallel" region, no dependencies in this "loop"
        //  computing a new solution
        for (int i = 1; i <= ROWS; i++)
            for (int j = 1; j <= COLUMNS; j++)
            {
                A_new[i][j] = 0.25 * (A[i + 1][j] + A[i - 1][j] + A[i][j + 1] + A[i][j - 1]);
            }

        dt = 0.0;

// Create a parallel region, specify the reduction operator and variable
// #pragma parallel loop reduction(max : dt)
#pragma parallel loop reduction(max \
                                : dt)
        //  updating the solution and finding the max error.
        for (int i = 1; i <= ROWS; i++)
            for (int j = 1; j <= COLUMNS; j++)
            {
                // dt can be modified by multiple worker, data race condition
                dt = fmax(fabs(A_new[i][j] - A[i][j]), dt);
                A[i][j] = A_new[i][j];
            }

        iteration++;
    }
}