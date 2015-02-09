#include <iostream>
#include "Kernels.h"
#define N 102400

int main(int argc, const char* argv[])
{
    int a[N];
    int b[N];
    
    for (int i = 0; i < N; i++) {
        a[i] = arc4random() % 100;
        b[i] = arc4random() % 100;
    }

    
    int* r;
    r = CUDAVectorAdd(a, b, N);
//    for (int i = 0; i < N; i++) {
//        std::cout << a[i] << ", ";
//    }
//    std::cout << std::endl;
//
//    for (int i = 0; i < N; i++) {
//        std::cout << b[i] << ", ";
//    }
//    std::cout << std::endl;
//
//    for (int i = 0; i < N; i++) {
//        std::cout << r[i] << ", ";
//    }

    delete [] r;

    
//    Matrix m1 = Matrix(3, 3);
//    Matrix m2 = Matrix(3, 3);
//    Matrix result = Matrix();
//    
//    m1.values = new float[ 3 * 3 ];
//    m2.values = new float[ 3 * 3 ];
//    
//
//    for (int i = 0; i < 3; i++) {
//        for (int j = 0; j < 3; j++) {
//            m1.values[i*3 + j] = arc4random() % 10;
//            m2.values[i*3 + j] = arc4random() % 10;
//        }
//    }
//    
//    result = CUDAMatrixMul(m1, m2);
//    
//    for (int i = 0; i < 3; i++) {
//        for (int j = 0; j < 3; j++) {
//            std::cout << m1.values[i*3 + j] << ", ";
//        }
//        std::cout << std::endl;
//    }
//    
//    std::cout << std::endl;
//
//    
//    for (int i = 0; i < 3; i++) {
//        for (int j = 0; j < 3; j++) {
//            std::cout << m2.values[i*3 + j] << ", ";
//        }
//        std::cout << std::endl;
//    }
//    
//    std::cout << std::endl;
//    
//
//    for (int i = 0; i < result.rows; i++) {
//        for (int j = 0; j < result.columns; j++) {
//            std::cout << result.values[i*3 + j] << ", ";
//        }
//        std::cout << std::endl;
//    }
//    
//    delete [] m1.values;
//    delete [] m2.values;
//    delete [] result.values;    
}
