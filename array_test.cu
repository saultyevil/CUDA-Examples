#include <stdio.h>

__global__
void array_op(int *device_array, int nx, int ny)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int colID = blockDim.x * blockIdx.x + tx;
    int rowID = blockDim.y * blockIdx.y + ty;


    if ((colID > 0 ) && (colID < nx-1) && (rowID > 0) && (rowID < ny-1))
    {
        printf("colID %d rowID %d\n", colID, rowID);
        device_array[rowID * nx + colID] = 69;
    }
}

int main(void)
{
    int i, nx, ny, size;
    int *host_array = NULL, *device_array = NULL;

    nx = 192;
    ny = 128;
    size = sizeof(int) * nx * ny;
    host_array = (int *) malloc(size);
    cudaMalloc((void **) &device_array, size);

    for(i = 0; i < nx * ny; i++)
        host_array[i] = 255;

    cudaMemcpy(device_array, host_array, size, cudaMemcpyHostToDevice);

    dim3 n_threads(8, 8);
    dim3 n_blocks(nx/n_threads.x + 1, ny/n_threads.y + 1);

    array_op<<<n_blocks, n_threads>>>(device_array, nx, ny);

    cudaMemcpy(host_array, device_array, size, cudaMemcpyDeviceToHost);

    FILE *f;
    f = fopen("output_array.txt", "w");

    for (i = 0; i < nx * ny; i++)
        fprintf(f, "%d\n", host_array[i]);

    fclose(f);

    return 0;

}