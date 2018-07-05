#include <stdio.h>

__global__ 
void CUDAhello(float f)
{
    printf("Hello from thread %d, f = %f\n", threadIdx.x, f);
}

int main(void)
{
    float f = 3.142;
    CUDAhello<<<1,10>>>(f);
    cudaDeviceReset();
    return 0;
}

