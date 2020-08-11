/* 
Game of Life using OpenACC

OpenACC CPU version, designed to use only the multicore CPU

Only use #pragma acc loop seq
*/

//#include <cuda.h>
#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h>  // for printf
#include <time.h>   // for nanosleep
#include <sys/time.h>

#define WIDTH 60
#define HEIGHT 40

void count_start(struct timeval *start);

double count_end(struct timeval *start, struct timeval *end);

const int offsets[8][2] = {{-1, 1}, {0, 1}, {1, 1}, {-1, 0}, {1, 0}, {-1, -1}, {0, -1}, {1, -1}};

void fill_board(int *board, int width, int height)
{
    int i;
    for (i = 0; i < width * height; i++)
        board[i] = rand() % 2;
}

void print_board(int *board, int width, int height)
{
    int x, y;

    printf("-------------------------------------------------------------\n");

    // for all rows
    for (y = 0; y < HEIGHT; ++y)
    {
        // for 1 row
        for (x = 0; x < WIDTH; ++x)
        {
            if (x >= width || y >= height)
            {
                printf("%c", ' ');
            }
            else
            {
                char c = board[y * width + x] ? '#' : ' ';
                printf("%c", c);
            }
        }
        printf("|\n");
    }

    printf("-------------------------------------------------------------\n");
}

void step(int *current, int *next, int width, int height)
{
    // coordinates of the cell we're currently evaluating
    int x, y;
    // offset index, neighbor coordinates, alive neighbor count
    int i, nx, ny, num_neighbors;

    // write the next board state
#pragma acc loop seq
    for (y = 0; y < height; y++)
    {
#pragma acc loop seq
        for (x = 0; x < width; x++)
        {
            // count this cell's alive neighbors
            num_neighbors = 0;
#pragma acc loop seq
            for (i = 0; i < 8; i++)
            {
                // To make the board torroidal, we use modular arithmetic to
                // wrap neighbor coordinates around to the other side of the
                // board if they fall off.
                nx = x + offsets[i][0];
                ny = y + offsets[i][1];

                if (nx >= 0 && ny >= 0 && nx < width && ny < height)
                {
                    nx = (nx + width) % width;
                    ny = (ny + height) % height;

                    if (current[ny * width + nx])
                    {
                        num_neighbors++;
                    }
                }
            }

            // apply the Game of Life rules to this cell
            next[y * width + x] = 0;
            if ((current[y * width + x] && num_neighbors == 2) ||
                num_neighbors == 3)
            {
                next[y * width + x] = 1;
            }
        }
    }
}

int main(int argc, const char *argv[])
{
    // parse the width and height command line arguments, if provided
    int width, height, iters, out;
    if (argc < 3)
    {
        printf("usage: life iterations 1=print");
        exit(1);
    }
    iters = atoi(argv[1]);
    out = atoi(argv[2]);
    if (argc == 5)
    {
        width = atoi(argv[3]);
        height = atoi(argv[4]);
        printf("Running %d iterations at %d by %d pixels using OpenACC (CPU).\n", iters, width, height);
    }
    else
    {
        width = WIDTH;
        height = HEIGHT;
    }

    struct timespec delay = {0, 125000000}; // 0.125 seconds
    struct timespec remaining;
    // The two boards
    int *current, *next, many = 0;
    float total_elapsed_time_ms = 0.0;
    struct timeval startT, endT;

    size_t board_size = sizeof(int) * width * height;
    current = (int *)malloc(board_size); // same as: int current[width * height];
    next = (int *)malloc(board_size);    // same as: int next[width *height];

    // Initialize the global "current".
    fill_board(current, width, height);

    while (many < iters)
    {
        many++;
        if (out == 1)
            print_board(current, width, height);

        count_start(&startT);

        //evaluate the `current` board, writing the next generation into `next`.
        step(current, next, width, height);

        total_elapsed_time_ms += (float)count_end(&startT, &endT);

        // copy the `next` to CPU and into `current` to be ready to repeat the process
        // Copy the next state, that step() just wrote into, to current state
        memcpy(current, next, board_size);

        // We sleep only because textual output is slow and the console needs
        // time to catch up. We don't sleep in the graphical X11 version.
        if (out == 1)
            nanosleep(&delay, &remaining);
    }

    // calculate a mean time toward N iterations
    printf("Mean time to calculate next generation of board: %f ms.\n", (total_elapsed_time_ms / iters));

    return 0;
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

    elapsedTime = (end->tv_sec - start->tv_sec) * 1000.0;    // sec difference to ms
    elapsedTime += (end->tv_usec - start->tv_usec) / 1000.0; // us difference to ms

    return elapsedTime;
}
