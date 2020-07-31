/*
Perform matrix multiplication on 2 2-D matrices using CPU
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

void count_start(struct timeval *start);
double count_end(struct timeval *start, struct timeval *end);

int main(int argc, char **argv)
{
    int aa = fmax(99, 88);
    int ab = abs(-6);
    printf("fmax (99, 88) = %d\n", aa);
    printf("abc (-6) = %d\n", ab);

    if (argc != 2)
    {
        printf("Usage:\n%s matrix-width\n", argv[0]);
        exit(1);
    }

    //dimension vars
    int width = atoi(argv[1]);
    int **MA, **MB, **MC;
    int dot_product = 0; // temp dot product
    int MA_value = 3;
    int MB_value = 2;

    // dynamically allocate mem
    MA = (int **)malloc(sizeof(*MA) * width);
    MB = (int **)malloc(sizeof(*MB) * width);
    MC = (int **)malloc(sizeof(*MC) * width);
    for (int i = 0; i < width; ++i)
    {
        MA[i] = (int *)malloc(sizeof(int) * width);
        MB[i] = (int *)malloc(sizeof(int) * width);
        MC[i] = (int *)malloc(sizeof(int) * width);
        for (int j = 0; j < width; ++j)
        {
            //fill matrices
            MA[i][j] = MA_value;
            MB[i][j] = MB_value;
        }
    }

    printf("Perform Matrix Multiplication on [%d x %d] x [%d x %d]\n", width, width, width, width);

    // --------------------------------- start timing --------------------------- //
    struct timeval start, end;
    count_start(&start);

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // for each dot product in C
            for (int k = 0; k < width; k++)
            {
                // row from A x col from B
                dot_product += MA[i][k] * MB[k][j];
            }
            MC[i][j] = dot_product;
            dot_product = 0;
        }
    }
    // --------------------------------- end timing --------------------------- //

    printf("Using CPU:\n   time to calculate: %f ms.\n", count_end(&start, &end));

    // Check for errors (all values should be 3.0f)
    printf("Checking for error...\n");
    int max_error = 0;
    int target_value = MA_value * MB_value * width;
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            max_error = fmax(max_error, abs(MC[i][j] - target_value));
        }
    }
    printf("Max error: %d \n\n", max_error);

    return 0;
}

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

    elapsedTime = (end->tv_sec - start->tv_sec) * 1000.0;    // sec difference to ms
    elapsedTime += (end->tv_usec - start->tv_usec) / 1000.0; // us difference to ms

    return elapsedTime;
}
