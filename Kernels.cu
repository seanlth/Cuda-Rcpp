#include <cuda.h>
#include "Kernels.h"


__global__ void vectorMul(int* a, int* b)
{
    //blockIdx is like the 'i' in the code above
    a[blockIdx.x] = a[blockIdx.x] / b[blockIdx.x];
}


int* CUDAVectorMul(int* a, int* b, int length)
{
    int N = 32;
    
    int size = N * sizeof( int );   //space for N 'ints'
    
    int* d_a;  //device pointer to a, points to memory on GPU
    int* d_b;  //device pointer to b, points to memory on GPU
    
    cudaMalloc(&d_a, size);  //make space in GPU memory for a
    cudaMalloc(&d_b, size);  //make space in GPU memory for b
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); //copy values in a to GPU memory
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); //copy values in b to GPU memory
    
    //call the kernel with N threads
    vectorMul<<<1, N>>>(d_a, d_b);
    
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost); //copy values back to CPU
    
    //print results
    for (int i = 0; i < N; i++) {
        std::cout << "result " << i << " is = " << a[i] << std::endl;
    }
    
    
    cudaFree( d_a );
    cudaFree( d_b );
    
    free(a);
    free(b);
    
    cudaDeviceReset();
}



__global__ void vectorDiv(int* a, int* b)
{
    //blockIdx is like the 'i' in the code above
    a[blockIdx.x] = a[blockIdx.x] / b[blockIdx.x];
}


int* CUDAVectorDiv(int* a, int* b, int length)
{
    int N = 32;
    
    int size = N * sizeof( int );   //space for N 'ints'
    
    int* d_a;  //device pointer to a, points to memory on GPU
    int* d_b;  //device pointer to b, points to memory on GPU
    
    cudaMalloc(&d_a, size);  //make space in GPU memory for a
    cudaMalloc(&d_b, size);  //make space in GPU memory for b
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); //copy values in a to GPU memory
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); //copy values in b to GPU memory
    
    //call the kernel with N threads
    vectorDiv<<<1, N>>>(d_a, d_b);
    
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost); //copy values back to CPU
    
    //print results
    for (int i = 0; i < N; i++) {
        std::cout << "result " << i << " is = " << a[i] << std::endl;
    }
    
    
    cudaFree( d_a );
    cudaFree( d_b );
    
    free(a);
    free(b);
    
    cudaDeviceReset();
}


__global__ void vectorSub(int* a, int* b)
{
    //blockIdx is like the 'i' in the code above
    a[blockIdx.x] = a[blockIdx.x] - b[blockIdx.x];
}


int* CUDAVectorSub(int* a, int* b, int length)
{
    int N = 32;
    
    int size = N * sizeof( int );   //space for N 'ints'
    
    int* d_a;  //device pointer to a, points to memory on GPU
    int* d_b;  //device pointer to b, points to memory on GPU
    
    cudaMalloc(&d_a, size);  //make space in GPU memory for a
    cudaMalloc(&d_b, size);  //make space in GPU memory for b
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); //copy values in a to GPU memory
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); //copy values in b to GPU memory
    
    //call the kernel with N threads
    vectorSub<<<1, N>>>(d_a, d_b);
    
    cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost); //copy values back to CPU
    
    //print results
    for (int i = 0; i < N; i++) {
        std::cout << "result " << i << " is = " << a[i] << std::endl;
    }
    
    
    cudaFree( d_a );
    cudaFree( d_b );
    
    free(a);
    free(b);
    
    cudaDeviceReset();
}


__global__ void vectorAdd(int* a, int* b, int length)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (id >= length) {
        return;
    }
    
    //blockIdx is like the 'i' in the code above
    a[id] = a[id] + b[id];
}


int* SERIALVectorAdd(int* a, int* b, int length)
{
    float time;
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    
    int* result = new int[length];
    
    for (int i = 0; i < length; i++) {
        result[i] = a[i] + b[i];
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Time:  %3.1f ms \n", time);
    return result;
}

int* CUDAVectorAdd(int* a, int* b, int length)
{
    int* result = new int[length];
    
    int size = length * sizeof( int );   //space for N 'ints'
    
    int* d_a;  //device pointer to a, points to memory on GPU
    int* d_b;  //device pointer to b, points to memory on GPU
    
    cudaMalloc(&d_a, size);  //make space in GPU memory for a
    cudaMalloc(&d_b, size);  //make space in GPU memory for b
    
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); //copy values in a to GPU memory
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); //copy values in b to GPU memory
    
    
    //time
    float time;
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    
    dim3 threadsPerBlock(32);
    dim3 blocksPerGrid(length / 32);

    
    //call the kernel with N threads
    vectorAdd<<<800, threadsPerBlock>>>(d_a, d_b);
    
    if (cudaSuccess != cudaGetLastError()) {
        std::cout << "Error calling kernel" << std::endl;
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Time:  %3.1f ms \n", time);
    
    cudaMemcpy(result, d_a, size, cudaMemcpyDeviceToHost); //copy values back to CPU
    
    cudaFree( d_a );
    cudaFree( d_b );
    
    //reset device, useful for debugging
    cudaDeviceReset();
    
    return result;
}

//very bad implementation
__global__ void matrixMul(Matrix m1, Matrix m2, Matrix m3)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    
    float accum = 0;
    for (int i = 0; i < m1.columns; i++) {
        accum += m1.values[ row * m1.columns + i ] * m2.values[ i * m2.columns + column ];
    }
    m3.values[ row * m3.columns + column ] = accum;
}


Matrix CUDAMatrixMul(Matrix h_A, Matrix h_B)
{
    Matrix result = Matrix(h_A.rows, h_B.columns);
    
    //initialise matrix A in device
    Matrix d_A = Matrix(h_A.columns, h_A.rows);
    long A_size = h_A.columns * h_A.rows * sizeof(float);
    cudaMalloc(&d_A.values, A_size);
    cudaMemcpy(d_A.values, h_A.values, A_size, cudaMemcpyHostToDevice);
    
    //initialise matrix B in device
    Matrix d_B = Matrix(h_B.columns, h_B.rows);
    long B_size = h_B.columns * h_B.rows * sizeof(float);
    cudaMalloc(&d_B.values, B_size);
    cudaMemcpy(d_B.values, h_B.values, B_size, cudaMemcpyHostToDevice);
    
    //initialise result in device
    Matrix d_result = Matrix(h_A.rows, h_B.columns);
    long result_size = h_A.rows * h_B.columns * sizeof(float);
    cudaMalloc(&d_result.values, result_size);
    result.values = new float[h_A.rows * h_B.columns];
    
    //probably will leave this to the user, no good way of figuring out the best launch configuration
    dim3 threadsPerBlock(1, 1);
    dim3 blocksPerGrid(h_B.columns / threadsPerBlock.x, h_A.rows / threadsPerBlock.y);
    
    matmul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_result);
    
    if (cudaSuccess != cudaGetLastError()) {
        std::cout << "Error calling kernel" << std::endl;
    }
    
    cudaMemcpy(result.values, d_result.values, result_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A.values);
    cudaFree(d_B.values);
    cudaFree(d_result.values);
    
    cudaDeviceReset();
    
    return result;
}



