#include <iostream>
#include "Kernels.h"

double* runif(int N)
{
    int d = 10000;
    
    double* r = new double[N];
    
    for (unsigned int i = 0; i < N; i++) {
        r[i] = (arc4random() % d) / d;
    }
    return r;
}


int main(int argc, const char* argv[])
{
    int number_of_iterations = atoi( argv[1] );
    int vector_length = 22;
    int data_length = 10000;
    
    double* data = runif(data_length);
    
    double** a = new double*[number_of_iterations];
    double** b = new double*[number_of_iterations];
    
    for (unsigned int i = 0; i < number_of_iterations; i++) {
        a[i] = runif(vector_length);
        b[i] = runif(vector_length);
    }
    
    double* result = new double[number_of_iterations];
    
    result = CUDALogLikelihood(data, data_length, a, b, vector_length, number_of_iterations);
    
    
    for (unsigned int i = 0; i < number_of_iterations; i++) {
        delete [] a[i];
        delete [] b[i];
        //std::cout << result[i] << std::endl;
    }
    
    delete [] a;
    delete [] b;

    delete [] data;
    delete [] result;
}
