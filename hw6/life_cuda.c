/* Compile with `gcc life.c`.
 * When CUDA-fied, compile with `nvcc life.cu`
 */

#include <cuda.h>
#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h>  // for printf
#include <time.h>   // for nanosleep

#define WIDTH 60
#define HEIGHT 40

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

int get_number_of_block(int thread_need, int block_size)
{
    return (thread_need + block_size - 1) / block_size;
}

// traditional host function (C style)
int allocate_device_mem(int **dev_arr, int N)
{
    cudaError_t err = cudaMalloc((void **)dev_arr, N);

    if (err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        return 0;
    }

    return 1;
}

// fill the board with random #
void fill_board(int *board, int width, int height)
{
    int i;
    int size = width * height;

    for (i = 0; i < size; i++)
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

__global__ void kernel_step(int *current, int *next, int width, int height)
{
    int offsets[8][2] = {
        {-1, 1},
        {0, 1},
        {1, 1},
        {-1, 0},
        {1, 0},
        {-1, -1},
        {0, -1},
        {1, -1}};

    // Calculate the row index of current board
    int y = blockIdx.y * blockDim.y + threadIdx.y; // "Row" in reg

    // Calculate the column index of current board
    int x = blockIdx.x * blockDim.x + threadIdx.x; // "Col" in reg

    if ((y < height) && (x < width))
    {
        // offset index, neighbor coordinates, alive neighbor count
        int i, nx, ny, num_neighbors;

        // write the next board state
        int cell_index = y * width + x;

        // count this cell's alive neighbors
        num_neighbors = 0;
        for (i = 0; i < 8; ++i)
        {
            // To make the board toroidal, we use modular arithmetic to
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

        // apply the Game of Life rules to the next generation cell
        // current access 1 time
        // next access 1 time
        if ((current[cell_index] && num_neighbors == 2) || num_neighbors == 3)
        {
            next[cell_index] = 1; // make this cell alive
        }
        else
        {
            next[cell_index] = 0; // make this cell dead
        }
    }
}

int main(int argc, const char *argv[])
{
    // parse the width and height command line arguments, if provided
    int width, height, iters, out;
    int block_dim_x;
    int block_dim_y;
    int grid_dim_x;
    int grid_dim_y;

    if (argc < 3)
    {
        printf("usage: ./a.out <iterations> <1=print> <width> <height>\n");
        exit(1);
    }

    iters = atoi(argv[1]);
    out = atoi(argv[2]);

    if (argc == 5)
    {
        width = atoi(argv[3]);
        height = atoi(argv[4]);

        if (width > 32 || height > 32)
        {
            // printf("Only accommodate square	board sizes	up to 32x32.\n");
            // exit(1);
            block_dim_x = 32;
            block_dim_y = 32;
            grid_dim_x = get_number_of_block(width, 32);
            grid_dim_y = get_number_of_block(height, 32);
        }
        else
        {
            block_dim_x = width;
            block_dim_y = height;
            grid_dim_x = 1;
            grid_dim_y = 1;
        }

        printf("Running %d iterations at %d by %d pixels using CUDA.\n", iters, width, height);
    }
    else
    {
        width = WIDTH;
        height = HEIGHT;
    }

    struct timespec delay = {0, 125000000}; // 0.125 seconds
    struct timespec remaining;

    // The two boards
    int *current,
        // *next,
        *dev_current,
        *dev_next,
        many = 0;

    size_t board_size = sizeof(int) * width * height;

    current = (int *)malloc(board_size); // same as: int current[width * height];
    // next = (int *)malloc(board_size);    // same as: int next[width *height];

    // Initialize the global "current".
    fill_board(current, width, height);

    // --------------------- using CUDA memory ------------------------ //

    // 1. Allocate device memory
    if (allocate_device_mem(&dev_current, board_size) == 0)
        exit(EXIT_FAILURE);

    if (allocate_device_mem(&dev_next, board_size) == 0)
        exit(EXIT_FAILURE);

    dim3 dimGrid(grid_dim_x, grid_dim_y, 1);
    dim3 dimBlock(block_dim_x, block_dim_y, 1);

    // --------------------- using CUDA memory ------------------------ //

    float total_elapsed_time_ms = 0.0;

    while (many < iters)
    {
        cudaEvent_t start, stop; // using cuda events to measure time
        float elapsed_time_ms;   // which is applicable for asynchronous code also

        many++;
        if (out == 1)
            print_board(current, width, height);

        // 2
        // copy data to device memory
        cudaMemcpy(dev_current, current, board_size, cudaMemcpyHostToDevice);

        // measure start time
        cuda_measure_start(&start, &stop);

        // 3
        // Kernel launch
        kernel_step<<<dimGrid, dimBlock>>>(dev_current, dev_next, width, height);

        // measure end time
        cuda_measure_stop(&start, &stop, &elapsed_time_ms);

        // 4
        // copy data from the device memory to host memory
        cudaMemcpy(current, dev_next, board_size, cudaMemcpyDeviceToHost);

        total_elapsed_time_ms += elapsed_time_ms;

        // We sleep only because textual output is slow and the console needs
        // time to catch up. We don't sleep in the graphical X11 version.
        if (out == 1)
            nanosleep(&delay, &remaining);
    }

    // calculate a mean time toward N iterations
    printf("Mean time to calculate next generation of board: %f ms.\n", (total_elapsed_time_ms / iters));

    // Free device vectors
    // cudaFree(pointer to freed obj)
    cudaFree(dev_current);
    cudaFree(dev_next);

    return 0;
}
