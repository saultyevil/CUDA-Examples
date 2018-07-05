#include <stdio.h>

#define CUDA_THREAD_ID_X blockDim.x * blockIdx.x + threadIdx.x
#define CUDA_THREAD_ID_Y blockDim.y * blockIdx.y + threadIdx.y

#define N 3


__global__
void CUDA_matrix_addition(int *m1, int *m2, int *m3)
{
    int colID = CUDA_THREAD_ID_X;
    int rowID = CUDA_THREAD_ID_Y;

    int index = rowID * N + colID;

    if (colID < N && rowID < N)
        m3[index] = m1[index] + m2[index];

}

int main(void)
{
    int i, j;
    dim3 grid(3, 3, 1);
    dim3 block(1, 1, 1);

    /*
     * Allocate memory for 2D arrays and initialise with some variables...
     */
    int *m1 = (int *) malloc(sizeof(*m1) * N * N);
    int *m2 = (int *) malloc(sizeof(*m2) * N * N);
    int *m_result = (int *) malloc(sizeof(*m_result) * N * N);

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            m1[i * N + j] = i;
            m2[i * N + j] = j;
        }
    }

    /*
     * Allocate the same as above but for CUDA memory and copy the results
     */
    int *CUDA_m1;
    int *CUDA_m2;
    int *CUDA_m_result;

    cudaMalloc((void **) &CUDA_m1, sizeof(*CUDA_m1) * N * N);
    cudaMalloc((void **) &CUDA_m2, sizeof(*CUDA_m2) * N * N);
    cudaMalloc((void **) &CUDA_m_result, sizeof(*CUDA_m_result) * N * N);

    cudaMemcpy(CUDA_m1, m1, sizeof(*m1) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(CUDA_m2, m2, sizeof(*m2) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(CUDA_m_result, m_result, sizeof(*m_result) * N * N, cudaMemcpyHostToDevice);

    CUDA_matrix_addition<<<grid, block>>>(CUDA_m1, CUDA_m2, CUDA_m_result);

    cudaMemcpy(m_result, CUDA_m_result, sizeof(*CUDA_m_result) * N * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            printf("m1[%d][%d] (%d) + m2[%d][%d] (%d) = %d\n",
                    i, j, m1[i * N + j],
                    i, j, m2[i * N + j],
                    m_result[i * N + j]);


    printf("Resulting matrix\n c = \n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", m_result[i*N+j]);
        }
        printf("\n");
    }

    free(m1); free(m2); free(m_result);
    cudaFree(CUDA_m1); cudaFree(CUDA_m2); cudaFree(CUDA_m_result);

    return 0;
}

