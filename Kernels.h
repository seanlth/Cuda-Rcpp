#ifndef _Kernels_h
#define _Kernels_h

#include <iostream>
#include <stdio.h>

class Matrix {
public:
    float* values;
    int columns;
    int rows;
    
    Matrix() {}
    
    Matrix(int columns, int rows) {
        this->columns = columns;
        this->rows = rows;
    }
};


//matrix operations
Matrix CUDAMatrixMul(Matrix h_A, Matrix h_B);

//vector operations
int* SERIALVectorAdd(int* a, int* b, int length);
int* CUDAVectorAdd(int* a, int* b, int length);
int* CUDAVectorSub(int* a, int* b, int length);
int* CUDAVectorDiv(int* a, int* b, int length);
int* CUDAVectorMul(int* a, int* b, int length);


double* CUDALogLikelihood(double* data, unsigned int data_length, double** a, double** b, unsigned int vector_length, unsigned int iterations);

#endif
