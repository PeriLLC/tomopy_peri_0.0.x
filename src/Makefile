CUDADIR ?= /usr/local/cuda

NVCC = $(CUDADIR)/bin/nvcc

TOMOPY_LIB_DIR ?= ~/PeriDev/peri_tomopy/tomopy_sandbox/lib/python2.7/site-packages/tomopy-0.0.3-py2.7-linux-x86_64.egg/tomopy/lib

all: pml_cuda_test

libpml_cuda.so: pml_cuda_kernel.o pml_cuda.o
	$(CC) -shared -o libpml_cuda.so pml_cuda_kernel.o pml_cuda.o -L$(CUDADIR)/lib64 -Wall -fPIC -lcuda -lcudart -lm -lstdc++ 

pml_cuda.o:
	$(CC) -c -o pml_cuda.o pml_cuda.c -Werror -fPIC -I$(CUDADIR)/include 

pml_cuda_kernel.o:
	$(NVCC) -c -arch=sm_20 pml_cuda_kernel.cu --compiler-options '-fPIC'

clean:
	rm -f *.o *.so *.dll *.exp *.obj *.lib

install: all
	cp libpml_cuda.so $(TOMOPY_LIB_DIR)

pml_cuda_test: libpml_cuda.so
	$(CC) -o pml_cuda_test pml_cuda_test.c -Wl,-rpath '-Wl,$$ORIGIN' -L. -L$(CUDADIR)/lib64 -lcuda -lcudart -lm -lpml_cuda 

test: pml_cuda_test
	./pml_cuda_test
