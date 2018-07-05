#include <stdio.h>
#include <cuda.h>
#define N 4096

#define HostToDevice cudaMemcpyHostToDevice
#define DeviceToHost cudaMemcpyDeviceToHost
#define CUDA_THREAD_INDEX blockIdx.x * blockDim.x + threadIdx.x

__global__
void vectorMulti(int *a, int *b, int *c)
{
    int threadID = CUDA_THREAD_INDEX;
    if (threadID < N)
        c[threadID] = a[threadID] * b[threadID];
}

int main(void)
{
    dim3 grid(64, 1, 1);
    dim3 block(64, 1, 1);

    int *a = (int *) malloc(sizeof(*a) * N);
    int *b = (int *) malloc(sizeof(*b) * N);
    int *c = (int *) malloc(sizeof(*c) * N);
    int *CUDA_a;
    int *CUDA_b;
    int *CUDA_c;
   
    cudaMalloc((void **) &CUDA_a, sizeof(*CUDA_a) * N); 
    cudaMalloc((void **) &CUDA_b, sizeof(*CUDA_b) * N); 
    cudaMalloc((void **) &CUDA_c, sizeof(*CUDA_c) * N); 

    /*
     * Initialise the data
     */
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    /*
     * Copy data to CUDA memory, call the kernel and then copy
     * the result back to the host memory
     */
    cudaMemcpy(CUDA_a, a, N * sizeof(*a), HostToDevice);
    cudaMemcpy(CUDA_b, b, N * sizeof(*b), HostToDevice);
    cudaMemcpy(CUDA_c, c, N * sizeof(*c), HostToDevice);
    vectorMulti<<<grid, block>>>(CUDA_a, CUDA_b, CUDA_c);
    cudaMemcpy(c, CUDA_c, N * sizeof(*CUDA_c), DeviceToHost);

    /*
     * Print the resulting vector
     */
    for (int i = 0; i < N; i++)
        printf("%d*%d=%d\n", a[i], b[i], c[i]); 

    free(a); free(b); free(c);
    cudaFree(CUDA_a); cudaFree(CUDA_b); cudaFree(CUDA_c);
    
    return 0;
}
