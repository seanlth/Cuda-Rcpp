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
    
    //blockIdx is like the 'id' in the code above
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
    vectorAdd<<<800, threadsPerBlock>>>(d_a, d_b, length);
    
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
    
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_result);
    
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

void serialLogLikelihood(double* a, double* b, double* data, unsigned int vector_length, unsigned int data_length, double* result)
{
    for (unsigned int i = 0; i < data_length; i++) {
        double sum = 0;
        for (unsigned int j = 0; j < vector_length; j++) {
            sum += exp(a[j] + b[j]*data[i]);
        }
        *result += log(sum);
    }
}


double* SERIALLogLikelihood(double* data, unsigned int data_length, double** a, double** b, unsigned int vector_length, unsigned int iterations)
{
    //time
    float time;
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    double* result = new double[iterations];
    double tmp[] = {0};
    
    for (unsigned int i = 0; i < iterations; i++) {
        
        double* a_temp = a[i];
        double* b_temp = b[i];

        serialLogLikelihood(a_temp, b_temp, data, vector_length, data_length, tmp);
        result[i] = *tmp;
    }
    
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    printf("%3.1f\n", time);
    
    return result;
}


__global__ void logLikelihood(double* a, double* b, double* data, unsigned int vector_length, unsigned int data_length, double* result)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    double sum = 0;
    for (unsigned int j = 0; j < vector_length; j++) {
        sum += exp(a[j] + b[j]*data[i]);
    }
    result[i] = log(sum);
    
}

double* CUDALogLikelihood(double* data, unsigned int data_length, double** a, double** b, unsigned int vector_length, unsigned int iterations)
{
    //time
    float time;
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    
    double* d_data; //device pointer for data
    unsigned int size_of_data = sizeof(double) * data_length;
    cudaMalloc(&d_data, size_of_data);  //make space in GPU memory for data
    cudaMemcpy(d_data, data, size_of_data, cudaMemcpyHostToDevice); //copy data in a to GPU memory
    
    double* d_a; //device pointers for a and b
    double* d_b;
    unsigned int size_of_args = sizeof(double) * vector_length;
    cudaMalloc(&d_a, size_of_args);  //make space in GPU memory for args
    cudaMalloc(&d_b, size_of_args);  //make space in GPU memory for args

    double* tmp = new double[data_length];
    double* d_tmp;
    cudaMalloc(&d_tmp, sizeof(double) * data_length);  //make space in GPU memory for result
    
    dim3 threadsPerBlock(32);
    
    int round_up = ( data_length + 32 - 1 ) / 32;
    
    dim3 blocksPerGrid( round_up );

    double* result = new double[iterations];
    
    for (unsigned int i = 0; i < iterations; i++) {
        
        double* a_temp = a[i];
        double* b_temp = b[i];
        
        cudaMemcpy(d_a, a_temp, size_of_args, cudaMemcpyHostToDevice); //copy 'a' to GPU memory
        cudaMemcpy(d_b, b_temp, size_of_args, cudaMemcpyHostToDevice); //copy 'b' to GPU memory

        logLikelihood<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_data, vector_length, data_length, d_tmp);
        
        cudaMemcpy(tmp, d_tmp, sizeof(double) * data_length, cudaMemcpyDeviceToHost);

        for (unsigned int j = 0; j < data_length; j++) {
            result[i] += tmp[j];
        }        
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_data);
    cudaFree(d_tmp);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    
    cudaDeviceReset();

    
    printf("%3.1f\n", time);
    
    return result;
}


