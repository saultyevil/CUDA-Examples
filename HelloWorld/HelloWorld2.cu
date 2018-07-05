#include <stdio.h>

__global__
void helloCUDA(float f)
{
    int thread = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Hello from block %d running on thread %d, f = %f\n",
            blockIdx.x, threadIdx.x, f);
}

int main(void)
{
    float f = 123.1231;
    helloCUDA<<<3, 10>>>(f);
    cudaDeviceReset();
    return 0;
}
