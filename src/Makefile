#    Copyright 2014-2015 Dake Feng, Peri LLC, dakefeng@gmail.com
#
#    This file is part of TomograPeri.
#
#    TomograPeri is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    TomograPeri is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with TomograPeri.  If not, see <http://www.gnu.org/licenses/>.

CUDADIR ?= /usr/local/cuda

NVCC = $(CUDADIR)/bin/nvcc

all: pml_cuda_test

libpml_cuda.so: pml_cuda_kernel.o pml_cuda.o
	$(CC) -shared -o libpml_cuda.so pml_cuda_kernel.o pml_cuda.o -L$(CUDADIR)/lib64 -Wall -fPIC -lcuda -lcudart -lm -lstdc++ 

pml_cuda.o:
	$(CC) -c -o pml_cuda.o pml_cuda.c -Werror -fPIC -I$(CUDADIR)/include 

pml_cuda_kernel.o:
	$(NVCC) -c -arch=sm_20 pml_cuda_kernel.cu --compiler-options '-fPIC'

clean:
	rm -f *.o *.so *.dll *.exp *.obj *.lib

pml_cuda_test: libpml_cuda.so
	$(CC) -o pml_cuda_test pml_cuda_test.c -Wl,-rpath '-Wl,$$ORIGIN' -L. -L$(CUDADIR)/lib64 -lcuda -lcudart -lm -lpml_cuda 

test: pml_cuda_test
	./pml_cuda_test
