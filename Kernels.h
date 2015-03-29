#ifndef _Kernels_h
#define _Kernels_h

#include <iostream>
#include <stdio.h>

double* CUDALogLikelihood(double* data, unsigned int data_length, double** a, double** b, unsigned int vector_length, unsigned int iterations);

#endif
