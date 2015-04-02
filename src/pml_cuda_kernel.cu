
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_functions.h"

#include "pml_cuda.h"
#include <stdio.h>

#define blockx 16
#define blocky 16


inline int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

typedef unsigned int  uint;

__global__ void _GEFrkernel_cuda(int num_slices,int num_grid,float* dev_recon,float* dev_G,float* dev_E,float* dev_F)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x;
    uint n = blockIdx.y*blockDim.y + threadIdx.y;
	uint k = blockIdx.z;
	uint i = m + n*num_grid + k*num_grid*num_grid;
	if((m>=num_grid)||(n>=num_grid)||(k>=num_slices))
		return;
//	int i = m + n*num_grid + k*num_grid*num_grid;
//  recon[i] = (-G[i]+sqrt(G[i]*G[i]-8*E[i]*F[i]))/(4*F[i]);
    dev_recon[i] = (-dev_G[i]+sqrtf(dev_G[i]*dev_G[i]-8.*dev_E[i]*dev_F[i]))/(4.*dev_F[i]);
}

__global__ void _sGkernel_cuda(int num_slices, int num_grid, float* dev_G,float *dev_suma)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x;
    uint n = blockIdx.y*blockDim.y + threadIdx.y;
	uint k = blockIdx.z;
	uint i = m + n*num_grid + k*num_grid*num_grid;
	uint j = m + n*num_grid;
	if((m>=num_grid)||(n>=num_grid)||(k>=num_slices))
		return;
//	G[k*num_grid*num_grid+n] += suma[n];
	dev_G[i]+= dev_suma[j];
}


__global__ void _weightInnerkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg8, float *dev_recon)
{
    uint m = blockIdx.x*blockDim.x + threadIdx.x+1;
    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;
	uint k = blockIdx.z;
	int q;
    int ind0, indg[8];

	if ((k>=num_slices)||(n<1)||(n>=(num_grid-1))||(m<1)||(m>=(num_grid-1)))
		return;

    ind0 = m + n*num_grid + k*num_grid*num_grid;
                    
    indg[0] = ind0+1;
    indg[1] = ind0-1;
    indg[2] = ind0+num_grid;
    indg[3] = ind0-num_grid;
    indg[4] = ind0+num_grid+1; 
    indg[5] = ind0+num_grid-1;
    indg[6] = ind0-num_grid+1;
    indg[7] = ind0-num_grid-1;
                    
    for (q = 0; q < 8; q++) {
        dev_F[ind0] += 2*beta*dev_wg8[q];
        dev_G[ind0] -= 2*beta*dev_wg8[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
    }
}

__global__ void _weightTopkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon)
{
	uint q;
    int ind0, indg[5];
    uint k = blockIdx.x*blockDim.x + threadIdx.x;
    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;

	if ((k>=num_slices)||(n<1)||(n>=(num_grid-1)))
		return;
    ind0 = n + k*num_grid*num_grid;
                
    indg[0] = ind0+1;
    indg[1] = ind0-1;
    indg[2] = ind0+num_grid;
    indg[3] = ind0+num_grid+1; 
    indg[4] = ind0+num_grid-1;
                    
    for (q = 0; q < 5; q++) {
        dev_F[ind0] += 2*beta*dev_wg5[q];
        dev_G[ind0] -= 2*beta*dev_wg5[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
    }
}

__global__ void _weightBottomkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon)
{
	uint q;
    int ind0, indg[5];
    uint k = blockIdx.x*blockDim.x + threadIdx.x;
    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;

	if ((k>=num_slices)||(n<1)||(n>=(num_grid-1)))
		return;

	ind0 = n + (num_grid-1)*num_grid + k*num_grid*num_grid;
                
    indg[0] = ind0+1;
    indg[1] = ind0-1;
    indg[2] = ind0-num_grid;
    indg[3] = ind0-num_grid+1;
    indg[4] = ind0-num_grid-1;
                    
    for (q = 0; q < 5; q++) {
        dev_F[ind0] += 2*beta*dev_wg5[q];
        dev_G[ind0] -= 2*beta*dev_wg5[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
    }
}


__global__ void _weightLeftkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon)
{
	uint q;
    int ind0, indg[5];
    uint k = blockIdx.x*blockDim.x + threadIdx.x;
    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;

	if ((k>=num_slices)||(n<1)||(n>=(num_grid-1)))
		return;

	ind0 = n*num_grid + k*num_grid*num_grid;
                
    indg[0] = ind0+1;
    indg[1] = ind0+num_grid;
    indg[2] = ind0-num_grid;
    indg[3] = ind0+num_grid+1; 
    indg[4] = ind0-num_grid+1;
                    
    for (q = 0; q < 5; q++) {
        dev_F[ind0] += 2*beta*dev_wg5[q];
        dev_G[ind0] -= 2*beta*dev_wg5[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
    }
}


__global__ void _weightRightkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon)
{
	uint q;
    int ind0, indg[5];
    uint k = blockIdx.x*blockDim.x + threadIdx.x;
    uint n = blockIdx.y*blockDim.y + threadIdx.y+1;

	if ((k>=num_slices)||(n<1)||(n>=(num_grid-1)))
		return;

    ind0 = (num_grid-1) + n*num_grid + k*num_grid*num_grid;
                
    indg[0] = ind0-1;
    indg[1] = ind0+num_grid;
    indg[2] = ind0-num_grid;
    indg[3] = ind0+num_grid-1;
    indg[4] = ind0-num_grid-1;
                    
    for (q = 0; q < 5; q++) {
        dev_F[ind0] += 2*beta*dev_wg5[q];
        dev_G[ind0] -= 2*beta*dev_wg5[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
    }
}


__global__ void _weightTLeftkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon)
{
	int ind0, indg[3],q;
	int k=blockIdx.x*blockDim.x + threadIdx.x;
	if (k>=num_slices)
		return;
    ind0 = k*num_grid*num_grid;
            
    indg[0] = ind0+1;
    indg[1] = ind0+num_grid;
    indg[2] = ind0+num_grid+1; 
                    
    for (q = 0; q < 3; q++) {
       dev_F[ind0] += 2*beta*dev_wg3[q];
       dev_G[ind0] -= 2*beta*dev_wg3[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
    }
}


__global__ void _weightTRightkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon)
{
	int ind0, indg[3],q;
	int k=blockIdx.x*blockDim.x + threadIdx.x;
	if (k>=num_slices)
		return;
    ind0 = (num_grid-1) + k*num_grid*num_grid;
            
    indg[0] = ind0-1;
    indg[1] = ind0+num_grid;
    indg[2] = ind0+num_grid-1;
                    
    for (q = 0; q < 3; q++) {
        dev_F[ind0] += 2*beta*dev_wg3[q];
        dev_G[ind0] -= 2*beta*dev_wg3[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
    }
}


__global__ void _weightBLeftkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon)
{
    int ind0, indg[3],q;
	int k=blockIdx.x*blockDim.x + threadIdx.x;
	if (k>=num_slices)
		return;
	ind0 = (num_grid-1)*num_grid + k*num_grid*num_grid;
            
	indg[0] = ind0+1;
	indg[1] = ind0-num_grid;
	indg[2] = ind0-num_grid+1;
                    
	for (q = 0; q < 3; q++) {
		dev_F[ind0] += 2*beta*dev_wg3[q];
		dev_G[ind0] -= 2*beta*dev_wg3[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
	}
}


__global__ void _weightBRightkernel_cuda(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon)
{
    int ind0, indg[3],q;
	int k=blockIdx.x*blockDim.x + threadIdx.x;
	if (k>=num_slices)
		return;
	ind0 = (num_grid-1) + (num_grid-1)*num_grid + k*num_grid*num_grid;
            
	indg[0] = ind0-1;
	indg[1] = ind0-num_grid;
	indg[2] = ind0-num_grid-1;
                    
    for (q = 0; q < 3; q++) {
        dev_F[ind0] += 2*beta*dev_wg3[q];
        dev_G[ind0] -= 2*beta*dev_wg3[q]*(dev_recon[ind0]+dev_recon[indg[q]]);
    }
}

__global__ void _kernelpp_cuda(int num_projections, float mov, int num_pixels, 
							 int num_grid, int num_slices, float* dev_gridx, float* dev_gridy, 
							 float* dev_suma, float * dev_E, 
							 float* dev_data, float * dev_recon, float* dev_theta){
    uint q = blockIdx.x*blockDim.x + threadIdx.x;
    uint m = blockIdx.y*blockDim.y + threadIdx.y;
    const double PI = 3.141592653589793238462;
	bool quadrant;
    float sinq, cosq;
	float xi, yi;
    float srcx, srcy, detx, dety;
    float slope, islope;
	int n,i,j,k;
    int alen, blen, len;
    int i1, i2;
    float x1, x2;
    int indx, indy;
    int io;
    float midx, midy, diffx, diffy;
    float simdata;
    float upd;
    float coordx[MAX_NUM_GRID];
    float coordy[MAX_NUM_GRID];
    float ax[MAX_NUM_GRID];
    float ay[MAX_NUM_GRID];
    float bx[MAX_NUM_GRID];
    float by[MAX_NUM_GRID];
    float coorx[MAX_NUM_GRID*2];
    float coory[MAX_NUM_GRID*2];
    float leng[MAX_NUM_GRID*2];
    int indi[MAX_NUM_GRID*2];
	if((m>=num_pixels)||(q>=num_projections))
		return;


    // Calculate the sin and cos values 
    // of the projection angle and find
    // at which quadrant on the cartesian grid.
    sinq = sin(dev_theta[q]);
    cosq =  cos(dev_theta[q]);
    if ((dev_theta[q] >= 0 && dev_theta[q] < PI/2) || 
            (dev_theta[q] >= PI && dev_theta[q] < 3*PI/2)) {
        quadrant = true;
    } else {
        quadrant = false;
    }

	// Find the corresponding source and
    // detector locations for a given line
    // trajectory of a projection (Projection
    // is specified by sinq and cosq). 
    xi = -1e6;
    yi = -(num_pixels-1)/2.+m+mov;
    srcx = xi*cosq-yi*sinq;
    srcy = xi*sinq+yi*cosq;
    detx = -xi*cosq-yi*sinq;
    dety = -xi*sinq+yi*cosq;
                
    // Find the intersection points of the 
    // line connecting the source and the detector
    // points with the reconstruction grid. The 
    // intersection points are then defined as: 
    // (coordx, gridy) and (gridx, coordy)
    slope = (srcy-dety)/(srcx-detx);
    islope = 1/slope;

    for (n = 0; n <= num_grid; n++) {
        coordx[n] = islope*(dev_gridy[n]-srcy)+srcx;
        coordy[n] = slope*(dev_gridx[n]-srcx)+srcy;
    }
                
    // Merge the (coordx, gridy) and (gridx, coordy)
    // on a single array of points (ax, ay) and trim
    // the coordinates that are outside the
    // reconstruction grid. 
    alen = 0;
    blen = 0;
    for (n = 0; n <= num_grid; n++) {
        if (coordx[n] > dev_gridx[0]) {
            if (coordx[n] < dev_gridx[num_grid]) {
                ax[alen] = coordx[n];
                ay[alen] = dev_gridy[n];
                alen++;
            }
        }
        if (coordy[n] > dev_gridy[0]) {
            if (coordy[n] < dev_gridy[num_grid]) {
                bx[blen] = dev_gridx[n];
                by[blen] = coordy[n];
                blen++;
            }
        }
    }
    len = alen+blen;
                
    // Sort the array of intersection points (ax, ay).
    // The new sorted intersection points are 
    // stored in (coorx, coory).
    i = 0;
    j = 0;
    k = 0;
    if (quadrant) {
        while (i < alen && j < blen)
        {
            if (ax[i] < bx[j]) {
                coorx[k] = ax[i];
                coory[k] = ay[i];
                i++;
                k++;
            } else {
                coorx[k] = bx[j];
                coory[k] = by[j];
                j++;
                k++;
            }
        }
        while (i < alen) {
            coorx[k] = ax[i];
            coory[k] = ay[i];
            i++;
            k++;
        }
        while (j < blen) {
            coorx[k] = bx[j];
            coory[k] = by[j];
            j++;
            k++;
        }
    } else {
        while (i < alen && j < blen)
        {
            if (ax[alen-1-i] < bx[j]) {
                coorx[k] = ax[alen-1-i];
                coory[k] = ay[alen-1-i];
                i++;
                k++;
            } else {
                coorx[k] = bx[j];
                coory[k] = by[j];
                j++;
                k++;
            }
        }
                    
        while (i < alen) {
            coorx[k] = ax[alen-1-i];
            coory[k] = ay[alen-1-i];
            i++;
            k++;
        }
        while (j < blen) {
            coorx[k] = bx[j];
            coory[k] = by[j];
            j++;
            k++;
        }
    }
                
    // Calculate the distances (leng) between the 
    // intersection points (coorx, coory). Find 
    // the indices of the pixels on the  
    // reconstruction grid (indi).
    for (n = 0; n < len-1; n++) {
        diffx = coorx[n+1]-coorx[n];
        diffy = coory[n+1]-coory[n];
        leng[n] = sqrt(diffx*diffx+diffy*diffy);
        midx = (coorx[n+1]+coorx[n])/2;
        midy = (coory[n+1]+coory[n])/2;
        x1 = midx+num_grid/2.;
        x2 = midy+num_grid/2.;
        i1 = (int)(midx+num_grid/2.);
        i2 = (int)(midy+num_grid/2.);
        indx = i1-(i1>x1);
        indy = i2-(i2>x2);
        indi[n] = indx+indy*num_grid;
    }
                
    // Note: The indices (indi) and the corresponding 
    // weights (leng) are the same for all slices. So,
    // there is no need to calculate them for each slice.
                

    //*******************************************************
    // Below is for updating the reconstruction grid.

    for (n = 0; n < len-1; n++) {
//        suma[indi[n]] += leng[n];
		atomicAdd(&(dev_suma[indi[n]]),leng[n]);
    }
                
    for (k = 0; k < num_slices; k++) {
        i = k*num_grid*num_grid;
        io = m + k*num_pixels + q*num_slices*num_pixels;
                    
        simdata = 0;
        for (n = 0; n < len-1; n++) {
            simdata += dev_recon[indi[n]+i] * leng[n];
        }
        upd = dev_data[io]/simdata;
        for (n = 0; n < len-1; n++) {
//            E[indi[n]+i] -= dev_recon[indi[n]+i]*upd*leng[n];
			atomicAdd(&(dev_E[indi[n]+i]),-dev_recon[indi[n]+i]*upd*leng[n]);
        }
    }
}

extern "C" {

	//! kernal function for updating recon from GEF
	cudaError_t GEFrkernelCUDA(int num_slices,int num_grid,float* dev_recon,float* dev_G,float* dev_E,float* dev_F)
	{
		cudaError_t cudaStatus=cudaSuccess;
		dim3 grid(iDivUp(num_grid,blockx),iDivUp(num_grid,blocky),num_slices);
		dim3 size(blockx,blocky);
	    
		_GEFrkernel_cuda<<<grid, size>>>(num_slices,num_grid, dev_recon, dev_G, dev_E, dev_F);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_GEFrkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching GEFrkernel!\n");
		}
		return cudaStatus;
	}
	
	cudaError_t sGkernelCUDA(int num_slices, int num_grid, float* dev_G,float *dev_suma)
	{
		cudaError_t cudaStatus=cudaSuccess;
		dim3 grid(iDivUp(num_grid,blockx),iDivUp(num_grid,blocky),num_slices);
		dim3 size(blockx,blocky);
	
	    _sGkernel_cuda<<<grid, size>>>(num_slices,num_grid, dev_G, dev_suma);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_sGkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching GEFrkernel!\n");
		}
		return cudaStatus;
	}
	
	
	
	cudaError_t  weightInnerCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg8, float *dev_recon)
	{
		cudaError_t cudaStatus=cudaSuccess;
		dim3 grid(iDivUp(num_grid-2,blockx),iDivUp(num_grid-2,blocky),num_slices);
		dim3 size(blockx,blocky);
	
	    _weightInnerkernel_cuda<<<grid, size>>>(num_slices, num_grid, beta, dev_F, dev_G, dev_wg8, dev_recon);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_weightInnerkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightInnerkernel_cuda!\n");
		}
		return cudaStatus;
	}
	
	cudaError_t  weightTopCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon)
	{
		cudaError_t cudaStatus=cudaSuccess;
		dim3 grid(iDivUp(num_slices,blockx),iDivUp(num_grid-2,blocky),1);
		dim3 size(blockx,blocky);
	
	    _weightTopkernel_cuda<<<grid, size>>>(num_slices, num_grid, beta, dev_F, dev_G, dev_wg5, dev_recon);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_weightTopkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightTopkernel_cuda!\n");
		}
		return cudaStatus;
	}
	
	cudaError_t  weightBottomCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon)
	{
		cudaError_t cudaStatus=cudaSuccess;
		dim3 grid(iDivUp(num_slices,blockx),iDivUp(num_grid-2,blocky),1);
		dim3 size(blockx,blocky);
	
	    _weightBottomkernel_cuda<<<grid, size>>>(num_slices, num_grid, beta, dev_F, dev_G, dev_wg5, dev_recon);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_weightBottomkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightBottomkernel_cuda!\n");
		}
		return cudaStatus;
	}
	
	cudaError_t  weightLeftCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon)
	{
		cudaError_t cudaStatus=cudaSuccess;
		dim3 grid(iDivUp(num_slices,blockx),iDivUp(num_grid-2,blocky),1);
		dim3 size(blockx,blocky);
	
	    _weightLeftkernel_cuda<<<grid, size>>>(num_slices, num_grid, beta, dev_F, dev_G, dev_wg5, dev_recon);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_weightLeftkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightLeftkernel_cuda!\n");
		}
		return cudaStatus;
	}
	
	cudaError_t  weightRightCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg5, float *dev_recon)
	{
		cudaError_t cudaStatus=cudaSuccess;
		dim3 grid(iDivUp(num_slices,blockx),iDivUp(num_grid-2,blocky),1);
		dim3 size(blockx,blocky);
	
	    _weightRightkernel_cuda<<<grid, size>>>(num_slices, num_grid, beta, dev_F, dev_G, dev_wg5, dev_recon);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_weightRightkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightRightkernel_cuda!\n");
		}
		return cudaStatus;
	}
	
	
	cudaError_t  weightTLeftCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon)
	{
		cudaError_t cudaStatus=cudaSuccess;
	
	    _weightTLeftkernel_cuda<<<iDivUp(num_slices,blockx), blockx>>>(num_slices, num_grid, beta, dev_F, dev_G, dev_wg3, dev_recon);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_weightTLeftkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightTLeftkernel_cuda!\n");
		}
		return cudaStatus;
	}
	
	
	cudaError_t  weightTRightCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon)
	{
		cudaError_t cudaStatus=cudaSuccess;
	
	    _weightTRightkernel_cuda<<<iDivUp(num_slices,blockx), blockx>>>(num_slices, num_grid, beta, dev_F, dev_G, dev_wg3, dev_recon);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_weightTRightkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightTRightkernel_cuda!\n");
		}
		return cudaStatus;
	
	}
	
	
	cudaError_t  weightBLeftCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon)
	{
		cudaError_t cudaStatus=cudaSuccess;
	
	    _weightBLeftkernel_cuda<<<iDivUp(num_slices,blockx), blockx>>>(num_slices, num_grid, beta, dev_F, dev_G, dev_wg3, dev_recon);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_weightBLeftkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightBLeftkernel_cuda!\n");
		}
		return cudaStatus;
	}
	
	
	cudaError_t  weightBRightCUDA(int num_slices, int num_grid, float beta, float *dev_F, float *dev_G, float*dev_wg3, float *dev_recon)
	{
		cudaError_t cudaStatus=cudaSuccess;
	
	    _weightBRightkernel_cuda<<<iDivUp(num_slices,blockx), blockx>>>(num_slices, num_grid, beta, dev_F, dev_G, dev_wg3, dev_recon);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_weightBRightkernel_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _weightBRightkernel_cuda!\n");
		}
		return cudaStatus;
	}
	
	
	cudaError_t  kernelppCUDA(int num_projections, float mov, int num_pixels, 
								 int num_grid, int num_slices, float* dev_gridx, float* dev_gridy, 
								 float* dev_suma, float * dev_E, 
								 float* dev_data, float * dev_recon, float* dev_theta){
		cudaError_t cudaStatus=cudaSuccess;
		dim3 grid(iDivUp(num_projections,blockx),iDivUp(num_pixels,blocky),1);
		dim3 size(blockx,blocky);
	
	    _kernelpp_cuda<<<grid, size>>>(num_projections, mov, num_pixels, 
								 num_grid, num_slices, dev_gridx, dev_gridy, 
								 dev_suma, dev_E, 
								 dev_data, dev_recon, dev_theta);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_kernelpp_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
		
	    // cudaDeviceSynchronize waits for the kernel to finish, and returns
	    // any errors encountered during the launch.
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _kernelpp_cuda!\n");
		}
		return cudaStatus;
	}
	
	__global__ void _kernel_clearsuma_cuda(int num_grid,float *dev_suma)
	{
	    uint m = blockIdx.x*blockDim.x + threadIdx.x;
	    uint n = blockIdx.y*blockDim.y + threadIdx.y;
		uint i = m + n*num_grid;
		if((m>=num_grid)||(n>=num_grid))
			return;
	    dev_suma[i] = 0.0;
	}
	
	__global__ void _kernel_clearsuna_EFG(int num_slices, int num_grid, float*dev_EFG)
	{
	    uint m = blockIdx.x*blockDim.x + threadIdx.x;
	    uint n = blockIdx.y*blockDim.y + threadIdx.y;
		uint k = blockIdx.z;
		uint i = m + n*num_grid + k*num_grid*num_grid;
		if((m>=num_grid)||(n>=num_grid)||(k>=num_slices))
			return;
	    dev_EFG[i] = 0.0;
	}
	
	cudaError_t  clearMemCUDA(int num_slices, int num_grid,float* dev_suma,float* dev_E,float* dev_F,float* dev_G)
	{
		cudaError_t cudaStatus=cudaSuccess;
		dim3 grid(iDivUp(num_grid,blockx),iDivUp(num_grid,blocky),1);
		dim3 size(blockx,blocky);
	    
		_kernel_clearsuma_cuda<<<grid, size>>>(num_grid, dev_suma);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_kernel_clearsuma_cuda launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
	
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _kernel_clearsuma_cuda!\n");
		}
	
		grid.z = num_slices;
	
		_kernel_clearsuna_EFG<<<grid, size>>>(num_slices,num_grid,dev_E);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_kernel_clearsuna_EFG launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
	
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _kernel_clearsuna_EFG!\n");
		}
	
	
		_kernel_clearsuna_EFG<<<grid, size>>>(num_slices,num_grid,dev_F);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_kernel_clearsuna_EFG launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
	
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _kernel_clearsuna_EFG!\n");
		}
	
		_kernel_clearsuna_EFG<<<grid, size>>>(num_slices,num_grid,dev_G);
	
	    // Check for any errors launching the kernel
	    cudaStatus = cudaGetLastError();
	    if (cudaStatus != cudaSuccess) {
	        fprintf(stderr, "_kernel_clearsuna_EFG launch failed: %s\n", cudaGetErrorString(cudaStatus));
	    }
	
		cudaStatus=cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code after launching _kernel_clearsuna_EFG!\n");
		}
	
		return cudaStatus;
	}
	

}