#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define tile_width 32

int CPU_matrix_multiply(float *m1, float *m2, float *m_r, int width)
{
    float partial_result = 0;

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            for (int k = 0; k < width; k++)
            {
                partial_result += m1[i * width + k] * m2[k * width + j];
            }
            m_r[i * width + j] = partial_result;
        }
    }

    return 0;
}

__global__
void CUDA_matrix_multiply_simple(float *m1, float *m2, float *m_r, int width)
{
    int rowID = blockIdx.y * blockDim.y + threadIdx.y;
    int colID = blockIdx.x * blockDim.x + threadIdx.x;
    float partial_result = 0;

    for (int i = 0; i < width; i++)
        partial_result += m1[rowID * width + i] * m2[i * width + colID];

    m_r[rowID * width + colID] = partial_result;
}

__global__
void CUDA_matrix_multiply_optim(float *m1, float *m2, float *m_r, int width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int rowID = blockIdx.y * blockDim.y + ty;
    int colID = blockIdx.x * blockDim.x + tx;
    float partial_result = 0;

    /*
     * Allocate 2D tiles in __shared__ memory
     */
    __shared__ float shared_m1[tile_width][tile_width];
    __shared__ float shared_m2[tile_width][tile_width];

    /*
     * Loop over the tiles of the input in phases
     */
    for (int p = 0; p < width/tile_width; p++)
    {
        shared_m1[ty][tx] = m1[rowID * width + (p * tile_width + tx)];
        shared_m2[ty][tx] = m2[(p * tile_width + ty) * width + colID];

        /*
         * Wait until all the data is loaded before allowing any thread
         * in this block to continue
         */
        __syncthreads();

        /*
         * Now do the dot product between the rows and columns of the matrices
         */
        for (int i = 0; i < tile_width; i++)
        {
            partial_result += shared_m1[ty][i] * shared_m2[i][tx];
        }
        /*
         * Synchronised the threads again
         */
        __syncthreads();
    }

    /*
     * Write out the final result
     */
    m_r[rowID * width + colID] = partial_result;
}

int main(void)
{
    int n = 1024;
    dim3 block(tile_width, tile_width);
    dim3 grid(n/block.x, n/block.y);
    float CPU_start, CPU_end, CPU_time, CUDA_simple_time, CUDA_optim_time;
    cudaEvent_t CUDA_simple_start, CUDA_simple_end;
    cudaEvent_t CUDA_optim_start, CUDA_optim_end;

    /*
     * Allocate memory for matrices on host
     */
    int matrix_size = n * n * sizeof(float);
    float *m1 = (float *) malloc(matrix_size);
    float *m2 = (float *) malloc(matrix_size);
    float *m_r = (float *) malloc(matrix_size);

    /*
     * Allocate memory for matrices on device
     */
    float *cuda_m1;
    float *cuda_m2;
    float *cuda_m_r;
    cudaMalloc((void **) &cuda_m1, matrix_size);
    cudaMalloc((void **) &cuda_m2, matrix_size);
    cudaMalloc((void **) &cuda_m_r, matrix_size);

    /*
     * Generate random input for host matrices and copy to device
     */
    for (int i = 0; i < n * n; i++)
    {
        m1[i] = (float) (rand()/RAND_MAX);
        m2[i] = (float) (rand()/RAND_MAX);
    }

    cudaMemcpy(cuda_m1, m1, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_m2, m2, matrix_size, cudaMemcpyHostToDevice);

    printf("Measuring the execution time for CPU and CUDA...\n");
    /*
     * Time how long it takes to compute matrix multi on the CPU
     */
    CPU_start = omp_get_wtime();
    CPU_matrix_multiply(m1, m2, m_r, n);
    CPU_end = omp_get_wtime();
    CPU_time = (CPU_end - CPU_start) * 1000;

    /*
     * Time how long it takes to compute matrix multi on simple kernel
     */
    cudaEventCreate(&CUDA_simple_start);
    cudaEventCreate(&CUDA_simple_end);
    cudaEventRecord(CUDA_simple_start, 0);
    CUDA_matrix_multiply_simple<<<grid, block>>>(cuda_m1, cuda_m2, cuda_m_r, n);
    cudaEventRecord(CUDA_simple_end, 0);
    cudaEventSynchronize(CUDA_simple_end);
    cudaEventElapsedTime(&CUDA_simple_time, CUDA_simple_start, CUDA_simple_end);

    /*
     * Time the optimised version
     */
    cudaEventCreate(&CUDA_optim_start);
    cudaEventCreate(&CUDA_optim_end);
    cudaEventRecord(CUDA_optim_start, 0);
    CUDA_matrix_multiply_optim<<<grid, block>>>(cuda_m1, cuda_m2, cuda_m_r, n);
    cudaEventRecord(CUDA_optim_end, 0);
    cudaEventSynchronize(CUDA_optim_end);
    cudaEventElapsedTime(&CUDA_optim_time, CUDA_optim_start, CUDA_optim_end);

    /*
     * Print the final timings
     */
    printf("\n================================\n");
    printf("             RESULTS            \n");
    printf("================================\n");
    printf("CPU execution time: %f ms\n", CPU_time);
    printf("Simple kernel execution time: %f ms (%6.2fx speedup)\n",
           CUDA_simple_time, CPU_time/CUDA_simple_time);
    printf("Optimised kernel execution time: %f ms (%6.2fx speedup)\n",
           CUDA_optim_time, CPU_time/CUDA_optim_time);
    printf("================================\n");

    free(m1); free(m2); free(m_r);
    cudaFree(cuda_m1); cudaFree(cuda_m2); cudaFree(cuda_m_r);

    return 0;
}
