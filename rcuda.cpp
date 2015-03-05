#include <Rcpp.h>
#include<stdio.h>

#include "Kernels.h"
using namespace Rcpp;

double** listTo2dDouble(Rcpp::List list)
{
  int length = list.size();
  double ** temp = new double*[length];
  
  for(int i=0; i<length; i++) {
    std::vector<double> test = as<std::vector<double> >(list[i]);
    int size = test.size();
    
    temp[i] = new double[size];
    for(int j=0; j<size; j++)
    {
      temp[i][j] = test[j];
    }
  }
  return temp;
}

RcppExport SEXP cudaLog(SEXP a1, SEXP b1, SEXP t1, SEXP numIter)
{
  int iterations = Rcpp::as<int>(numIter);
  Rcpp::List a(a1);
  Rcpp::List b(b1);
  int len = as<std::vector<double> >(a[0]).size();
  
  double ** cA = listTo2dDouble(a);
  double ** cB = listTo2dDouble(b);
  Rcpp::NumericVector data(t1);
  double* res = CUDALogLikelihood(data.begin(), data.size(), cA, cB, len, iterations);
  
  std::vector<double> result(res, res + iterations);
  
  return Rcpp::wrap(result);
}
