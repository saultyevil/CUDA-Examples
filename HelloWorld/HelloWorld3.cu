#include <stdio.h>

__global__
void helloCUDA(float f)
{
    int block_id = blockIdx.x + blockIdx.y * gridDim.x;
    int thread_id = block_id * blockDim.y * blockDim.x + threadIdx.x + threadIdx.y * blockDim.x;
    printf("Hello from block %d (x %d y %d) running thread %d (x %d y %d), f = %f\n",
          block_id, blockIdx.x, blockIdx.y, thread_id, threadIdx.x, threadIdx.y, f);
}

int main(void)
{
    float f = 123.1231;
    dim3 grid(2,2,1);
    dim3 block(2,2,1);
    helloCUDA<<<grid, block>>>(f);
    cudaDeviceReset();
    return 0;
}
