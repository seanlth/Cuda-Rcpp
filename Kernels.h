#ifndef _Kernels_h
#define _Kernels_h

#include <iostream>


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


Matrix CUDAMatrixMul(Matrix h_A, Matrix h_B);

int* SERIALVectorAdd(int* a, int* b, int length);

int* CUDAVectorAdd(int* a, int* b, int length);
int* CUDAVectorSub(int* a, int* b, int length);
int* CUDAVectorDiv(int* a, int* b, int length);
int* CUDAVectorMul(int* a, int* b, int length);

#endif
