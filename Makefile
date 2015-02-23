NVCC= nvcc
FLAGS= -G -g -deviceemu -arch sm_20 -o R_CUDA 
all: R_CUDA
R_CUDA: main.cpp
	$(NVCC) main.cpp Kernels.cu $(FLAGS)
