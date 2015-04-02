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