/*
self-created sin function

compile:
gcc -Wall sin.c -o sin -lm

*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>

long double factorial_fun(int x)
{
    long double sum = 1;
    int i = 1;
    for (i = 1; i <= x; i++)
    {
        sum = sum * i;
    }

    return sum;
}

long double power_fun(double x, int y)
{
    long double sum = 1;
    int i;
    for (i = 1; i <= y; i++)
    {
        sum = sum * x;
    }

    return sum;
}

/*
Using taylor series
*/
double sin_fun(double z)
{
    int i = 1;
    long double value, val2, val3, sum = 0;
    for (i = 1; i < 33; i += 2)
    {
        val2 = power_fun(z, i);
        val3 = factorial_fun(i);
        value = val2 / val3;

        if (((i - 1) / 2) % 2 != 0)
        {
            sum = sum - value; //((power_fun(x,i))/factorial_fun(i));
        }
        else
        {
            sum = sum + value;
        }
    }

    // printf("\n%f\n", sum);
    return sum;
}
int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        printf("Usage %s x\n", argv[0]);
        exit(1);
    }

    double x = atof(argv[1]);

    printf("sin_fun( %f ) = %f \n", x, sin_fun(x));
    printf("sin( %f ) = %f \n", x, sin(x));

    printf("famx(5, 7) = %f", fmax(5, 7)); // need <math.h>
    printf("\nabs(5) = %d", abs(5));       // need <math.h>
    printf("\nabs(-5) = %d", abs(-5));     // need <math.h>
}