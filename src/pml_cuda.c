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


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pml_cuda.h"

cudaError_t initCuda(int num_projections, int num_slices, int num_pixels, int num_grid,float**pdev_suma,float**pdev_E,float**pdev_F,float**pdev_G,
					 float**pdev_gridx,float**pdev_gridy,float **pdev_data, float **pdev_theta, float** pdev_recon,float** pdev_wg8,float**pdev_wg5,float**pdev_wg3,
					 float* data, float* theta, float* recon,float *gridx, float *gridy);
void cleanCuda(float*dev_suma,float*dev_E,float*dev_F,float*dev_G,
			   float*dev_gridx,float*dev_gridy,float *dev_data, float *dev_theta, float* dev_recon, float* dev_wg8, float* dev_wg5, float* dev_wg3);

cudaError_t GEFrkernelCUDA(int num_slices,int num_grid,float* dev_recon,float* dev_G,float* dev_E,float* dev_F);

cudaError_t sGkernelCUDA(int num_slices, int num_grid, float* dev_G,float *dev_suma);

cudaError_t  weightInnerCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg8, float *dev_recon);
cudaError_t  weightTopCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon);
cudaError_t  weightBottomCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon);
cudaError_t  weightLeftCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon);
cudaError_t  weightRightCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon);
cudaError_t  weightTLeftCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon);
cudaError_t  weightTRightCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon);
cudaError_t  weightBLeftCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon);
cudaError_t  weightBRightCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon);

cudaError_t  kernelppCUDA(int num_projections, float mov, int num_pixels, 
						  int num_grid, int num_slices, float* dev_gridx, float* dev_gridy, 
						  float* dev_suma, float * dev_E, 
						  float* dev_data, float * dev_recon, float* dev_theta);

cudaError_t  clearMemCUDA(int num_slices, int num_grid, float* dev_suma,float* dev_E,float* dev_F,float* dev_G);

cudaError_t weightAllCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float* dev_recon,
				   float* dev_wg8, float*dev_wg5,float*dev_wg3);


/*! pml_cuda function.
 *
 *  @param data		array of float, index is defined as
 *					m + k*num_pixels + q*num_slices*num_pixels
 *					q: 0..num_projections; k:0..num_slices; m:0..num_pixels
 *					Therefore it is array with three dimensions, defined as
 *					float data[num_projections][num_slices][num_pixels];
 *
 *  @param theta	array of float, index is defined as
 *					q
 *					q: 0..num_projections;
 *					Therefore it is one dimension array, defined as
 *					float theta[num_projections]
 *
 *	@param center	float pixels, will be cleared like num_pixels/2-center;
 *
 *	@param num_projections
 *					int, number of projections
 *
 *	@param num_slices	
 *					int, number of slices
 *
 *	@param num_pixels
 *					int, number of pixels
 *
 *	@param num_grid	int, number of grids. This gives a size of grids for 
 *					recon, the grid area is square grids area. It will:
 *					define gridx, gridy, coordx, coordy, ax, ay, bx, by as float[num_grid+1];
 *					define coorx, coory, leng as float[2*num_grid];
 *					define indi as int[2*num_grid];
 *
 *	@param iter		int, number of iterations
 *
 *	@param beta		float, magic number
 *
 *	@param recon	array of float, index is defined as
 *					m + n*num_grid + k*num_grid*num_grid;
 *					m: 0..num_grid;n: 0..num_grid;k: 0..num_slices;
 *					Therefore it is array with three dimensions, defined as
 *					float recon[num_slices][num_grid][num_grid];
 *					
 *
 */
int pml_cuda(float* data, float* theta, float center, 
			  int num_projections, int num_slices, int num_pixels, 
			  int num_grid, int iters, float beta, float* recon) {
	
    int m, t;
    float mov;
	
    float* gridx;
    float* gridy;
	
	
	float* dev_suma;
    float* dev_E;
    float* dev_F;
    float* dev_G;
    float* dev_gridx;
    float* dev_gridy;
	
	float* dev_data, *dev_theta, *dev_recon;
	float* dev_wg8, *dev_wg5, *dev_wg3;
	int ret=PERI_CALL_SUCCESS;
	
	if (num_grid>MAX_NUM_GRID)
	{
		fprintf(stderr,"This version only support maximum num_grid %d, please increase it if you need more\n",MAX_NUM_GRID);
		return PERI_CALL_FAIL;
	}
	
    gridx = (float *)malloc((num_grid+1)*sizeof(float));
    gridy = (float *)malloc((num_grid+1) * sizeof(float));

	mov = num_pixels/2-center;
	if (mov-ceil(mov) < 1e-6) {
		mov += 1e-6;
	}
    
	// Define the reconstruction grid lines.
	for (m = 0; m <= num_grid; m++) {
		gridx[m] = -num_grid/2.+m;
		gridy[m] = -num_grid/2.+m;
	}
	
	if(initCuda(num_projections, num_slices, num_pixels, num_grid, &dev_suma,&dev_E,&dev_F,&dev_G,
				&dev_gridx,&dev_gridy,&dev_data, &dev_theta,&dev_recon,&dev_wg8,&dev_wg5,&dev_wg3,
				data, theta, recon,gridx,gridy)==cudaSuccess)
	{
		
		
		// For each iteration
		for (t = 0; t < iters; t++) {
			printf ("pml_cuda iteration: %i \n", t+1);
			
			
			if(clearMemCUDA(num_slices, num_grid, dev_suma,dev_E,dev_F,dev_G)!=cudaSuccess)
			{
				ret=PERI_CALL_FAIL;
				break;
			};

			//kernelpp
			if(kernelppCUDA(num_projections, mov, num_pixels, 
						 num_grid, num_slices, dev_gridx, dev_gridy, 
						 dev_suma, dev_E, 
						 dev_data, dev_recon, dev_theta)!=cudaSuccess)
			{
				ret=PERI_CALL_FAIL;
				break;
			};
			   
			//weightAll
			if(weightAllCUDA(num_slices, num_grid, beta, dev_F, dev_G, dev_recon,dev_wg8,dev_wg5,dev_wg3)!=cudaSuccess)
			{
				ret=PERI_CALL_FAIL;
				break;
			};
			
			
			//sGkernel
			if(sGkernelCUDA(num_slices,num_grid,dev_G,dev_suma)!=cudaSuccess)
			{
				ret=PERI_CALL_FAIL;
				break;
			};
			
			//GEFrkernel
			if(GEFrkernelCUDA(num_slices,num_grid,dev_recon,dev_G,dev_E,dev_F)!=cudaSuccess)
			{
				ret=PERI_CALL_FAIL;
				break;
			};
			
		}
		if (ret==PERI_CALL_SUCCESS){
			// Copy output vector from GPU buffer to host memory.
			if (cudaMemcpy(recon, dev_recon, (num_slices*num_grid*num_grid) * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed! recon\n");
			}
			else {
				ret=PERI_CALL_FAIL;
			}
		}
	}
	else {
		ret=PERI_CALL_FAIL;
	}

	
	cleanCuda(dev_suma,dev_E,dev_F,dev_G,dev_gridx,dev_gridy,dev_data, dev_theta, dev_recon,dev_wg8,dev_wg5,dev_wg3);
	free(gridx);
	free(gridy);
	return ret;
}


// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
typedef struct
{
	int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
	int Cores;
} sSMtoCores;

//! Beginning of GPU Architecture definitions
int _ConvertSMVer2Cores(int major, int minor)
{
	
    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
        { 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
        { 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
        { 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        {   -1, -1 }
    };
	
    int index = 0;
	
    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }
		
        index++;
    }
	
    // If we don't find the values, we default use the previous one to run properly
	//    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

int findCudaDevice(){
	int deviceId=-1;
	int deviceCount;
	int max_multiprocessors = 0;
	int i;
	struct cudaDeviceProp deviceProp;
	if(cudaGetDeviceCount(&deviceCount)!=cudaSuccess)
	{
		fprintf(stderr,"cudaGetDeviceCount() failed\n");
		return -1;
	}
	for (i=0;i<deviceCount;i++)
	{
		if (cudaGetDeviceProperties(&deviceProp,i)!=cudaSuccess)
		{
			fprintf(stderr,"cudaGetDeviceProperties(%d) failed\n",i);
			return -1;
		}
		
		printf("INFO: Found Cuda Device %d, name is %s\n",i,deviceProp.name);

		//sm>=2.0 is required!
		if((deviceProp.major<2)||(deviceProp.computeMode==cudaComputeModeProhibited))
			continue;
		if (max_multiprocessors < _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount) {
			max_multiprocessors = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
			deviceId = i;
		}
		
	}
	return deviceId;
}

int test_cuda()
{
	// Choose which GPU to run on, based on the fastest qulified GPU
	int deviceid=findCudaDevice();
	int ret=PERI_CALL_FAIL;
	cudaError_t cudaStatus=cudaErrorInitializationError;
    if (deviceid <0) {
        fprintf(stderr, "findCudaDevice failed!\nDo you have a CUDA-capable GPU with at least sm2.0 support and available?\n");
    }
	else{
		cudaStatus = cudaSetDevice(deviceid);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!\n");
		}
		else {
			printf("cudaDevice test ok!\n");
			ret=PERI_CALL_SUCCESS;
		}

	}
	cudaDeviceReset();
	return ret;
}

cudaError_t initCuda(int num_projections, int num_slices, int num_pixels, int num_grid, float**pdev_suma,float**pdev_E,float**pdev_F,float**pdev_G,
					 float**pdev_gridx,float**pdev_gridy,float **pdev_data, float **pdev_theta, float** pdev_recon,float** pdev_wg8,float**pdev_wg5,float**pdev_wg3,
					 float* data, float* theta, float* recon, float *gridx, float *gridy){
	
	cudaError_t cudaStatus=cudaErrorInitializationError;
	float *devptr;	
	float wg8[8],wg5[5],wg3[3];
	int deviceid=-1;

	* pdev_suma=NULL;
    * pdev_E=NULL;
    * pdev_F=NULL;
    * pdev_G=NULL;
    * pdev_gridx=NULL;
    * pdev_gridy=NULL;
	* pdev_data=NULL;
	* pdev_theta=NULL;
	* pdev_recon=NULL;
	* pdev_wg8=NULL;
	* pdev_wg5=NULL;
	* pdev_wg3=NULL;

	// Weights for inner neighborhoods.
    wg8[0] = 0.1464466094;
    wg8[1] = 0.1464466094;
    wg8[2] = 0.1464466094;
    wg8[3] = 0.1464466094;
    wg8[4] = 0.10355339059;
    wg8[5] = 0.10355339059;
    wg8[6] = 0.10355339059;
    wg8[7] = 0.10355339059;
	
    // Weights for edges.
    wg5[0] = 0.226540919667;
    wg5[1] = 0.226540919667;
    wg5[2] = 0.226540919667;
    wg5[3] = 0.1601886205;
    wg5[4] = 0.1601886205;
	
    // Weights for corners.
    wg3[0] = 0.36939806251;
    wg3[1] = 0.36939806251;
    wg3[2] = 0.26120387496;
	
	deviceid=findCudaDevice();
    // Choose which GPU to run on, based on the fastest qulified GPU
    if (deviceid <0) {
        fprintf(stderr, "findCudaDevice failed!\nDo you have a CUDA-capable GPU with at least sm2.0 support and available?\n");
        goto Error;
    }
	
    cudaStatus = cudaSetDevice(deviceid);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!\n");
        goto Error;
    }
	
	printf("INFO: Cuda Device %d is actived!\n",deviceid);

	/*
	size_t heapSize;
	cudaDeviceGetLimit ( &heapSize,cudaLimitMallocHeapSize);
	printf("original heap Size :%ld\n",heapSize);
	cudaDeviceSetLimit ( cudaLimitMallocHeapSize,heapSize*20);
	cudaDeviceGetLimit ( &heapSize,cudaLimitMallocHeapSize);
	printf("required new heap Size :%ld\n",heapSize);
	 */
	//	cudaDeviceReset();
	//	exit(0);
	
	
    // Allocate GPU buffers 
	//suma = (float *)calloc((num_grid*num_grid), sizeof(float));
    if (cudaMalloc((void**)&devptr, (num_grid*num_grid) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_suma=devptr;
	
	//E = (float *)calloc((num_slices*num_grid*num_grid), sizeof(float));
    if (cudaMalloc((void**)&devptr, (num_slices*num_grid*num_grid) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_E=devptr;
	
	//F = (float *)calloc((num_slices*num_grid*num_grid), sizeof(float));
    if (cudaMalloc((void**)&devptr, (num_slices*num_grid*num_grid) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_F=devptr;
	
	//G = (float *)calloc((num_slices*num_grid*num_grid), sizeof(float));
    if (cudaMalloc((void**)&devptr, (num_slices*num_grid*num_grid) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_G=devptr;
	
    //gridx = (float *)malloc((num_grid+1)*sizeof(float));
    if (cudaMalloc((void**)&devptr, (num_grid+1) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_gridx=devptr;
	
	//float* gridy = (float *)malloc((num_grid+1) * sizeof(float));
    if (cudaMalloc((void**)&devptr, (num_grid+1) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_gridy=devptr;
	
	//float* data
    if (cudaMalloc((void**)&devptr, (num_projections*num_slices*num_pixels) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_data=devptr;
	
	//float* theta
    if (cudaMalloc((void**)&devptr, (num_projections) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_theta=devptr;
	
	//float* recon
    if (cudaMalloc((void**)&devptr, (num_slices*num_grid*num_grid) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_recon=devptr;
	
	//float* wg8
    if (cudaMalloc((void**)&devptr, (8) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_wg8=devptr;
	
	//float* wg5
    if (cudaMalloc((void**)&devptr, (5) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_wg5=devptr;
	
	//float* wg3
    if (cudaMalloc((void**)&devptr, (3) * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
	*pdev_wg3=devptr;
	
	cudaStatus = cudaMemcpy(*pdev_data, data, (num_projections*num_slices*num_pixels) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(*pdev_theta, theta, (num_projections) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(*pdev_recon, recon, (num_slices*num_grid*num_grid) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(*pdev_wg8, wg8, (8) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(*pdev_wg5, wg5, (5) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(*pdev_wg3, wg3, (3) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(*pdev_gridx, gridx, (num_grid+1) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
    cudaStatus = cudaMemcpy(*pdev_gridy, gridy, (num_grid+1) * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	
	printf("INFO: Memory on Device %d allocated and copied!\n",deviceid);
	
Error:
	return cudaStatus;
}


void cleanCuda(float*dev_suma,float*dev_E,float*dev_F,float*dev_G,
			   float*dev_gridx,float*dev_gridy,float *dev_data, float *dev_theta, float* dev_recon,
			   float* dev_wg8, float* dev_wg5, float* dev_wg3){
	cudaError_t cudaStatus;
	
    cudaFree(dev_suma);
    cudaFree(dev_E);
    cudaFree(dev_F);
    cudaFree(dev_G);
    cudaFree(dev_gridx);
    cudaFree(dev_gridy);
    cudaFree(dev_data);
    cudaFree(dev_theta);
    cudaFree(dev_recon);
	cudaFree(dev_wg8);
	cudaFree(dev_wg5);
	cudaFree(dev_wg3);
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
    }
	else
		printf("INFO: Cuda device reset!\n");
	
};


/*! weightAll grid components, 9 kernels involved
 *
 */
cudaError_t weightAllCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float* dev_recon,
				   float* dev_wg8, float*dev_wg5,float*dev_wg3){
	cudaError_t status=cudaSuccess;
	// (inner region)
	status=weightInnerCUDA(num_slices, num_grid, beta, dev_F, dev_G, dev_wg8, dev_recon);
	if (status!=cudaSuccess) {
		return status;
	}
	
    // (top)
	status=weightTopCUDA(num_slices, num_grid, beta, dev_F, dev_G, dev_wg5, dev_recon);
	if (status!=cudaSuccess) {
		return status;
	}
    // (bottom)
	status=weightBottomCUDA(num_slices, num_grid, beta, dev_F, dev_G, dev_wg5, dev_recon);
	if (status!=cudaSuccess) {
		return status;
	}
    // (left)  
	status=weightLeftCUDA(num_slices, num_grid, beta, dev_F, dev_G, dev_wg5, dev_recon);
	if (status!=cudaSuccess) {
		return status;
	}
    // (right)                
	status=weightRightCUDA(num_slices, num_grid, beta, dev_F, dev_G, dev_wg5, dev_recon);
	if (status!=cudaSuccess) {
		return status;
	}
	
	
    // (top-left)
	status=weightTLeftCUDA(num_slices, num_grid, beta, dev_F, dev_G, dev_wg3, dev_recon);
	if (status!=cudaSuccess) {
		return status;
	}
    // (top-right)
	status=weightTRightCUDA(num_slices, num_grid, beta, dev_F, dev_G, dev_wg3, dev_recon);
	if (status!=cudaSuccess) {
		return status;
	}
    // (bottom-left)
	status=weightBLeftCUDA(num_slices, num_grid, beta, dev_F, dev_G, dev_wg3, dev_recon);
	if (status!=cudaSuccess) {
		return status;
	}
	// (bottom-right)        
	status=weightBRightCUDA(num_slices, num_grid, beta, dev_F, dev_G, dev_wg3, dev_recon);
	if (status!=cudaSuccess) {
		return status;
	}
	
	return status;
}
