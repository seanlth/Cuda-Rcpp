#include <Rcpp.h>
#include<stdio.h>
#include "Kernels.h"
using namespace Rcpp;

// Function to convert a Rcpp:List of vectors to a 2d double array
double** listTo2dDouble(Rcpp::List list)
{
  // Get length of list and allocate size of result array
  int length = list.size();
  double ** temp = new double*[length];
  
  for(int i=0; i<length; i++) {
    // Retrieve the vector at position i from the list
    std::vector<double> test = as<std::vector<double> >(list[i]);
    
	// Allocate space for the result array
	int size = test.size();
    temp[i] = new double[size];
	
	// Copy elements from vector to a double array
    for(int j=0; j<size; j++)
    {
      temp[i][j] = test[j];
    }
  }
  
  // Return the 2d double array 
  return temp;
}

// This is the exported Rcpp function that can be called from R
RcppExport SEXP cudaLog(SEXP a1, SEXP b1, SEXP t1, SEXP numIter)
{
  // Convert parameters from SEXP objects
  int iterations = Rcpp::as<int>(numIter);
  Rcpp::List a(a1);
  Rcpp::List b(b1);
  Rcpp::NumericVector data(t1);
  
  // Length of a vector from a or b (Should be the same)
  int len = as<std::vector<double> >(a[0]).size();
  
  // Convert a and b from a list of vectors to a 2d double array
  double ** cA = listTo2dDouble(a);
  double ** cB = listTo2dDouble(b);

  // Call the CUDA implementation of the log-likelihood function from Kernels.h with signature:
  // double* CUDALogLikelihood(double* data, unsigned int data_length, double** a, double** b, unsigned int vector_length, unsigned int iterations)
  double* res = CUDALogLikelihood(data.begin(), data.size(), cA, cB, len, iterations);
  
  // Convert the resultant double array to a double vector
  std::vector<double> result(res, res + iterations);
  
  // Wrap the resultant vector to a R object and return it
  return Rcpp::wrap(result);
}
