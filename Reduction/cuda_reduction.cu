#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int CPU_reduction_sum(int *data, int length)
{
    int sum = 0;
    for (int i = 0; i < length; i++)
        sum += data[i];

    return sum;
}

__global__
void cuda_reduction_sum(int *data, int *sum, int length)
{
    int tID = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tID;

    /*
     * We can define the amount of memory needing to be allocated when we call
     * the kernel in the host function
     */
    extern __shared__ int share_data[];
    share_data[tID]  = 0.0;

    if (i < N)
    {
        /*
         * Load the data into shared arrays
         */
        share_data[tID] = data[i];
        __synthreads();

        for (int i = 1; i < blockDim.x; i *= 2)
        {
            if (tID % (2 * i) == 0)
                share_data[tID] += share_data[tID + s];

            __syncthreads();
        }
        if (tID == 0)
            output[blockIdx.x] = share_data[0];
    }
}

int main(void)
{
    int N = 50000000; // number of array elements

    /*
     * Get the number of threads from device properties and then create the
     * block and grid dimension variables
     */
    cudaDeviceProp device_properties;
    cudaGetDeviceProperties(&device_properties, 0);
    int n_threads = device_properties.maxThreadsPerBlock;
    dim3 grid((N + n_threads -1)/n_threads, 1, 1);
    dim3 block(n_threads, 1, 1);

    printf("========================\n");
    printf("Single-GPU reduction sum\n");
    printf("========================\n");
    printf("Total array elements to sum: %d\n", N);
    printf("Kenel launch config: %d blocks of %d threads\n", grid.x, block.x);
    printf("========================\n");

    /*
     * Create some random data
     */
    int *data = (int *) malloc(N * sizeof(data));
    int *cuda_data;
    int *cuda_sum;
    cudaMalloc((void **) &cuda_data, N * sizeof(cuda_data));
    cudaMalloc((void **) &cuda_sum, grid.x * sizeof(cuda_sum));

    srand(time(NULL));
    for (int i = 0; i < N; i++)
        data[i] = rand()%10;

    cudaMemcpy(cuda_data, data, N * sizeof(cuda_data), cudaMemcpyHostToDevice);

    printf("\nCalculating reduction sum execution time on CPU and CUDA.\n");

    /*
     * Measure the CPU execution time
     */
    float CPU_start, CPU_end, CPU_time;
    CPU_start = omp_get_wtime();
    CPU_sum = CPU_reduction_sum(data, N);
    CPU_end = omp_get_wtime();
    CPU_time = (CPU_end - CPU_start) * 1000;

    /*
     * Measure the CUDA execution time
     */
    float cuda_time;
    cudaEvent_t cuda_start, cuda_end;
    cudaEventCreate(&cuda_start);
    cudaEventCreate(&cuda_end);
    cudaEventRecord(cuda_start, 0);
    CUDA_reduction_sum<<<grid, block, block.x * sizeof(int)>>>(cuda_data,
            cuda_sum, N);
    cudaEventRecord(cuda_end, 0);
    cudaEventSyncrhonize(cuda_end);
    cudaEventElapsedTime(&cuda_time, cuda_start, cuda_stop);

    printf("========================\n");
    printf("Execution time results:\n");
    printf("========================\n");
    printf("CPU execution time: %f ms. Sum = %d\n", CPU_time, CPU_sum);
    printf("CUDA execution time: %f ms (%dx speedup). Sum = %d\n", cuda_time,
           CPU_time/cuda_time, cuda_sum);

    return 0;
}

