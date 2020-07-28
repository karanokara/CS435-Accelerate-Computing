/**
 * In-class exercise 
 * 
 */

#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h>  // for printf
#include <time.h>   // for nanosleep
#include <sys/time.h>

// 1. Rewrite this code as a single loop nest
int main()
{
    int a[n][m], b[n][m], c[n][m];

    init(a, b, n, m);

    for (int j = 0; j < n; ++j)
    {
        for (int k = 0; k < m; ++k)
        {
            c[j][k] = a[j][k];
            a[j][k] = c[j][k] + b[j][k];
            d[j][k] = a[j][k] - 5;

            // d[j][k] = a[j][k] + b[j][k] - 5;
        }
    }
}

int main()
{
    int a[n][m], b[n][m], c[n][m];

    init(a, b, n, m);

#pragma acc kernels
    {

        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < m; ++k)
            {
                c[j][k] = a[j][k];
                a[j][k] = c[j][k] + b[j][k];
            }
        }

        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < m; ++k)
            {
                d[j][k] = a[j][k] - 5;
            }
        }
    }
}

int main()
{
    int a[n][m], b[n][m], c[n][m];

    init(a, b, n, m);

#pragma acc data create(c) create(d) copy(a[:n][:m]) copy(b[:n][:m])
    {
#pragma acc parallel loop
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < m; ++k)
            {
                c[j][k] = a[j][k];
                a[j][k] = c[j][k] + b[j][k];
            }
        }

#pragma acc parallel loop
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < m; ++k)
            {
                d[j][k] = a[j][k] - 5;
            }
        }
    }
}
