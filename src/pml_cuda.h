/*
    Copyright 2014-2015 Dake Feng, Peri LLC, dakefeng@gmail.com

    This file is part of TomograPeri.

    TomograPeri is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    TomograPeri is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with TomograPeri.  If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef PML_CUDA_SHARE_H
#define PML_CUDA_SHARE_H

#define MAX_NUM_GRID 256

#define PERI_CALL_SUCCESS 1
#define PERI_CALL_FAIL 0

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32	
#define DLL __declspec(dllexport) 
#else
#define DLL
#endif
	
int DLL pml_cuda(float* data, float* theta, float center, 
				  int num_projections, int num_slices, int num_pixels, 
				  int num_grid, int iters, float beta, float* recon);

int DLL test_cuda();
	
#ifdef __cplusplus
}
#endif

#endif  // PML_CUDA_SHARE_H
