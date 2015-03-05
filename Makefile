R_HOME := $(shell R RHOME)
R_INC := /usr/share/R/include
R_LIB := $(R_HOME)/lib

CUDA_HOME := /usr/local/cuda-6.5
CUDA_INC := $(CUDA_HOME)/include
CUDA_LIB := $(CUDA_HOME)/lib64

RCPP_HOME := ../R/x86_64-pc-linux-gnu-library/3.0/Rcpp
RCPP_INC := $(RCPP_HOME)/include

CPICFLAGS := $(shell R CMD config CPICFLAGS)

OBJS := rcuda.o Kernels.o

INCS := -I. -I"$(CUDA_INC)" -I"$(R_INC)" -I"$(RCPP_INC)"
PARAMS := -Xcompiler $(CPICFLAGS)

LD_PARAMS := -Xlinker -rpath,$(CUDA_LIB)
LIBS :=  -L$(R_LIB) -L$(CUDA_LIB) -lcublas -lcudart $(shell R CMD config BLAS_LIBS)

TARGETS := rcuda.so

NVCC := /usr/local/cuda-6.5/bin/nvcc

all: $(TARGETS) 

$(TARGETS): $(OBJS)
	$(NVCC) -shared $(LD_PARAMS) $(LIBS) $(OBJS) -o $@

Kernels.o: Kernels.cu
	$(NVCC) -c $(INCS) $(PARAMS) $^ -o $@

rcuda.o: rcuda.cpp
	$(NVCC) -c $(INCS) $(PARAMS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean