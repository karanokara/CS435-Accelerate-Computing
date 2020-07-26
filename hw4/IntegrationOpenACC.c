/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   : 
 To build use  : g++ -fopenmp integration_seq.cpp -o integration_seq
 ============================================================================
 */

// Sequential integration of testf
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

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

//---------------------------------------
double testf(double x)
{
    return x * x + 2 * sin(x);
}

//---------------------------------------
double integrate(double st, double en, int div, double (*f)(double))
{
    double localRes = 0;
    double step = (en - st) / div;
    double x;

    x = st;
    localRes = f(st) + f(en);
    localRes /= 2;

    for (int i = 1; i < div; i++)
    {
        x += step;
        localRes += f(x);
    }

    localRes *= step;

    return localRes;
}

//---------------------------------------
double integrate_openacc(double st, double en, int div, double (*f)(double))
{
    double localRes = 0;
    double step = (en - st) / div;
    double x;

    x = st;
    localRes = f(st) + f(en);
    localRes /= 2;

#pragma acc parallel loop reduction(+ \
                                    : localRes)
    for (int i = 1; i < div; i++)
    {
        x = step * i;
        localRes = x * x + 2 * sin(x);
        // localRes = testf(x);
    }

    localRes *= step;

    return localRes;
}

//---------------------------------------
int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        printf("Usage %s start end divisions\n", argv[0]);
        exit(1);
    }

    double start, end, finalRes;
    int divisions;

    start = atof(argv[1]);
    end = atof(argv[2]);
    divisions = atoi(argv[3]);

    struct timeval startT, endT;

    count_start(&startT);
    finalRes = integrate(start, end, divisions, testf);
    printf("Result: %f    Time elapsed using CPU: %f ms.\n", finalRes, count_end(&startT, &endT));

    count_start(&startT);
    finalRes = integrate_openacc(start, end, divisions, testf);
    printf("Result: %f    Time elapsed using openacc: %f ms.\n", finalRes, count_end(&startT, &endT));

    return 0;
}
