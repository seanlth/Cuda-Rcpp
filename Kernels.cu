#include <cuda.h>
#include "Kernels.h"

__global__ void logLikelihood2(double* a, double* b, double* data, unsigned int vector_length, unsigned int data_length, double* result)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    double sum = 0;
    for (unsigned int j = 0; j < data_length; j++) {
        sum += data[j];
    }
    
    result[i] = -b[i] * sum + data_length * (log(a[i]) + log(b[i])) - a[i] * (1 - exp(-b[i] * data[data_length-1]));
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
    std::fill(result, result+iterations, 0);
    
    for (unsigned int i = 0; i < iterations; i++) {
        
        double* a_temp = a[i];
        double* b_temp = b[i];
        
        cudaMemcpy(d_a, a_temp, size_of_args, cudaMemcpyHostToDevice); //copy 'a' to GPU memory
        cudaMemcpy(d_b, b_temp, size_of_args, cudaMemcpyHostToDevice); //copy 'b' to GPU memory

        logLikelihood2<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_data, vector_length, data_length, d_tmp);
        
        cudaMemcpy(tmp, d_tmp, sizeof(double) * data_length, cudaMemcpyDeviceToHost);

        //replace for CUDA horizontal sum
        for (unsigned int j = 0; j < data_length; j++) {
            result[i] += tmp[j];
        }
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_data);
    cudaFree(d_tmp);
    
    cudaDeviceReset();
    
    return result;
}


