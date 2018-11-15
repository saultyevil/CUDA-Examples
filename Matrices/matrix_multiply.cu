#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define TILE_WIDTH 32
#define CUFLOAT float

struct timespec get_time (void) {
  struct timespec time;

  clock_gettime (CLOCK_PROCESS_CPUTIME_ID, &time);

  return time;
}

CUFLOAT get_duration (struct timespec start_time) {
  CUFLOAT td;
  struct timespec end_time;

  clock_gettime (CLOCK_PROCESS_CPUTIME_ID, &end_time);
  td = (end_time.tv_sec - start_time.tv_sec) +
                                (end_time.tv_nsec - start_time.tv_nsec) * 1e-9;

  return td;
}

int CPU_matrix_multiply(CUFLOAT *m1, CUFLOAT *m2, CUFLOAT *m_r, int width)
{
    CUFLOAT partial_result = 0;

    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < width; k++) {
                partial_result += m1[i * width + k] * m2[k * width + j];
            }
            m_r[i * width + j] = partial_result;
        }
    }

    return 0;
}

__global__ void
simple_CUDA_matrix_multiply (CUFLOAT *m1, CUFLOAT *m2, CUFLOAT *m_r, int width) {
    int rowID = blockIdx.y * blockDim.y + threadIdx.y;
    int colID = blockIdx.x * blockDim.x + threadIdx.x;
    CUFLOAT partial_result = 0;

    for (int i = 0; i < width; i++)
        partial_result += m1[rowID * width + i] * m2[i * width + colID];

    m_r[rowID * width + colID] = partial_result;
}

__global__ void
optimised_CUDA_matrix_multiply (CUFLOAT *m1, CUFLOAT *m2, CUFLOAT *m_r, int width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int rowID = blockIdx.y * blockDim.y + ty;
    int colID = blockIdx.x * blockDim.x + tx;
    CUFLOAT partial_result = 0;

    /*
     * Allocate 2D tiles in __shared__ memory
     */

    __shared__ CUFLOAT shared_m1[TILE_WIDTH][TILE_WIDTH];
    __shared__ CUFLOAT shared_m2[TILE_WIDTH][TILE_WIDTH];

    /*
     * Loop over the tiles of the input in phases
     */

    for (int p = 0; p < width/TILE_WIDTH; p++) {
        shared_m1[ty][tx] = m1[rowID * width + (p * TILE_WIDTH + tx)];
        shared_m2[ty][tx] = m2[(p * TILE_WIDTH + ty) * width + colID];

        /*
         * Wait until all the data is loaded before allowing any thread
         * in this block to continue
         */

        __syncthreads();

        /*
         * Now do the dot product between the rows and columns of the matrices
         */

        for (int i = 0; i < TILE_WIDTH; i++)
            partial_result += shared_m1[ty][i] * shared_m2[i][tx];

        /*
         * Synchronise the threads again
         */

        __syncthreads();
    }

    /*
     * Write out the final result
     */

    m_r[rowID * width + colID] = partial_result;
}

int main (void) {
    int n = 1024;
    float CPU_time, CU_simple_time, CU_optimised_time;  // these have to be floats :-(

    struct timespec cpu_start;

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(n / block.x, n / block.y);

    cudaEvent_t CU_simple_start, CU_simple_end;
    cudaEvent_t CU_optim_start, CU_optim_end;

    /*
     * Allocate memory for matrices on host
     */

    int matrix_size = n * n * sizeof (CUFLOAT);
    CUFLOAT *m1 = (CUFLOAT *) malloc (matrix_size);
    CUFLOAT *m2 = (CUFLOAT *) malloc (matrix_size);
    CUFLOAT *m_r = (CUFLOAT *) malloc (matrix_size);

    /*
     * Allocate memory for matrices on device
     */

    CUFLOAT *cuda_m1;
    CUFLOAT *cuda_m2;
    CUFLOAT *cuda_m_r;
    cudaMalloc ((void **) &cuda_m1, matrix_size);
    cudaMalloc ((void **) &cuda_m2, matrix_size);
    cudaMalloc ((void **) &cuda_m_r, matrix_size);

    /*
     * Generate random input for host matrices and copy to device
     */

    for (int i = 0; i < n * n; i++) {
        m1[i] = (CUFLOAT) (rand () / RAND_MAX);
        m2[i] = (CUFLOAT) (rand () / RAND_MAX);
    }

    cudaMemcpy (cuda_m1, m1, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy (cuda_m2, m2, matrix_size, cudaMemcpyHostToDevice);

    printf ("Beginning CPU test\n");

    cpu_start = get_time ();
    CPU_matrix_multiply (m1, m2, m_r, n);
    CPU_time = get_duration (cpu_start);
    CPU_time *= 1000; // turn to ms

    /*
     * Naive GPU implementation
     */

    printf ("Beginning simple GPU test\n");

    cudaEventCreate (&CU_simple_start); cudaEventCreate (&CU_simple_end);
    cudaEventRecord (CU_simple_start, 0);

    simple_CUDA_matrix_multiply <<<grid, block>>> (cuda_m1, cuda_m2,
                                                   cuda_m_r, n);

    cudaEventRecord (CU_simple_end, 0);
    cudaEventSynchronize (CU_simple_end);
    cudaEventElapsedTime (&CU_simple_time, CU_simple_start, CU_simple_end);

    /*
     * Optimised GPU implementation
     */

    printf ("Beginning optimised GPU test\n");

    cudaEventCreate (&CU_optim_start); cudaEventCreate (&CU_optim_end);
    cudaEventRecord (CU_optim_start, 0);

    optimised_CUDA_matrix_multiply <<<grid, block>>> (cuda_m1, cuda_m2,
                                                      cuda_m_r, n);

    cudaEventRecord (CU_optim_end, 0);
    cudaEventSynchronize (CU_optim_end);
    cudaEventElapsedTime (&CU_optimised_time, CU_optim_start, CU_optim_end);

    printf ("\n================================\n");
    printf ("             RESULTS            \n");
    printf ("================================\n");
    printf ("CPU execution time: %f ms\n", CPU_time);
    printf ("Simple execution time: %f ms (%6.2fx speedup)\n",
            CU_simple_time, CPU_time / CU_simple_time);
    printf ("Optimised execution time: %f ms (%6.2fx speedup)\n",
            CU_optimised_time, CPU_time / CU_optimised_time);
    printf ("================================\n");

    free (m1); free (m2); free (m_r);
    cudaFree (cuda_m1); cudaFree (cuda_m2); cudaFree (cuda_m_r);

    return 0;
}
