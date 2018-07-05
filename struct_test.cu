#include <stdlib.h>
#include <stdio.h>

#define N 5

typedef struct Struct
{
    int *int_array;
    double *double_array;
} Struct;

__global__
void __struct_kernel(Struct test_struct)
{
    for (int i = 0; i < N; i++)
    {
        test_struct.int_array[i] = (i+1);
        test_struct.double_array[i] = 3.14*(i+1);
    }
}

int init_host(void)
{
    Struct *test_struct;
    Struct CUDA_test_struct;

    /*
     * Allocate and initialise struct on the host
     */
    test_struct = (Struct *) malloc(sizeof(Struct));
    test_struct->int_array = (int *) malloc(sizeof(int) * N);
    test_struct->double_array = (double *) malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++)
    {
        test_struct->int_array[i] = 0;
        test_struct->double_array[i] = 0;
    }

    /*
     * Now allocate memory for the variables on the device
     */
    cudaMalloc((void **) &CUDA_test_struct, sizeof(Struct));
    cudaMalloc((void **) &CUDA_test_struct.int_array, sizeof(int) * N);
    cudaMalloc((void **) &CUDA_test_struct.double_array, sizeof(double) * N);

    cudaMemcpy(CUDA_test_struct.int_array, test_struct->int_array,
                sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(CUDA_test_struct.double_array, test_struct->double_array,
                 sizeof(double) * N, cudaMemcpyHostToDevice);

    printf("\n======================\n");
    printf(" Arrays before kernel\n");
    printf("======================\n");

    for (int i = 0; i < N; i++)
        printf("int: %d\tdouble: %f\n", test_struct->int_array[i],
            test_struct->double_array[i]);

    /*
     * Call the CUDA kernel and copy the result
     */
    __struct_kernel<<<1, 1>>>(CUDA_test_struct);

    cudaMemcpy(test_struct->int_array, CUDA_test_struct.int_array,
                sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(test_struct->double_array, CUDA_test_struct.double_array,
               sizeof(double) * N, cudaMemcpyDeviceToHost);

    printf("\n=====================\n");
    printf(" Arrays after kernel\n");
    printf("=====================\n");
    for (int i = 0; i < N; i++)
        printf("int: %d\tdouble: %f\n", test_struct->int_array[i],
               test_struct->double_array[i]);

    /*
     * Free the memory
     */
    free(test_struct->int_array);
    free(test_struct->double_array);
    free(test_struct);

    return 0;
}

int main(void)
{
    init_host();
    return 0;
}
