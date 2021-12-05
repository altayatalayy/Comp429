/*

	Implement your CUDA kernel in this file

*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#include "cardiacsim_kernels.cuh"

//#define __constant__
//#define __global__

__constant__ double n;
__constant__ double m;
__constant__ double kk;
__constant__ double dt;
__constant__ double epsilon;
__constant__ double M1;
__constant__ double M2;
__constant__ double b;
__constant__ double a;
__constant__ double alpha;


void check2(cudaError_t err, const char * file, int line){
    if(err != cudaSuccess){
        printf("Cuda error occured, %s at line %d\nName:%s\nDescription:%s\n",\
            file, line, cudaGetErrorName(err), cudaGetErrorString(err));
    }
}
#undef __DEBUG__
#define __DEBUG__

#ifdef __DEBUG__
#define CUCall(x) (x); check2(cudaGetLastError(), __FILE__, __LINE__)
#else
#define CUCall(x) (x)
#endif

__global__ void simulate_v1_PDE(double *E, double *R, double *E_prev){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	//int tmp2 = tid - i;
	int tmp = 200; //n

	if (i>0 && j > 0 && (i < n+2 ) && (j < n+2)){
		 E[tid] = E_prev[tid] + alpha*(E_prev[tid-tmp]+E_prev[tid+tmp]-4*E_prev[tid]+E_prev[tid+1]+E_prev[tid-1]); 
	}
}
__global__ void simulate_v1_ODE(double *E, double *R, double *E_prev){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	int tid = blockIdx.x * blockDim.x + threadIdx.x;


	if (i>0 && j> 0 && (i < n+2 ) && (j < n+2)){
		E[tid] = E[tid] -dt*(kk* E[tid]*(E[tid] - a)*(E[tid]-1)+ E[tid] *R[tid]);
		R[tid] = R[tid] + dt*(epsilon+M1* R[tid]/( E[tid]+M2))*(-R[tid]-kk* E[tid]*(E[tid]-b-1));
	}
    
}    
/*
__global__ void simulate_v2(double *E, double *R, double *E_prev){

  	int i = blockIdx.x*blockDim.x + threadIdx.x; 
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	if ((i < n+2 ) && (j < n+2)){
		 E[j][i] = E_prev[j][i]+alpha*(E_prev[j][i+1]+E_prev[j][i-1]-4*E_prev[j][i]+E_prev[j+1][i]+E_prev[j-1][i]);
		 E[j][i] = E[j][i] -dt*(kk* E[j][i]*(E[j][i] - a)*(E[j][i]-1)+ E[j][i] *R[j][i]);
		 R[j][i] = R[j][i] + dt*(epsilon+M1* R[j][i]/( E[j][i]+M2))*(-R[j][i]-kk* E[j][i]*(E[j][i]-b-1));
	}
    
}
*/

__global__ void simulate_v3(double *E, double *R, double *E_prev){
	int j = blockIdx.x*blockDim.x + threadIdx.x; 
	int i = blockIdx.y*blockDim.y + threadIdx.y;

	int grid_width = gridDim.x * blockDim.x;
    int index = i * grid_width + j;


	if (i>0 && j> 0 && (i < n+2 ) && (j < n+2)){
		E[index] = E[index] -dt*(kk* E[index]*(E[index] - a)*(E[index]-1)+ E[index] *R[index]);
		R[index] = R[index] + dt*(epsilon+M1* R[index]/( E[index]+M2))*(-R[index]-kk* E[index]*(E[index]-b-1));
	}
}

namespace Simulation{
	double *h_E, *h_E_prev, *h_R;
	double *d_E, *d_E_prev, *d_R;

	int hn, hm;

	void init(double **E, double **E_prev, double **R, int h_alpha, int h_n, int h_m, double h_a,double h_kk, double h_dt, double h_epsilon, double h_M1, double h_M2, double h_b){
		hn = h_n; hm = h_m;
		//init consts
		CUCall(cudaMemcpyToSymbol(n, &h_n, sizeof(int)));
		CUCall(cudaMemcpyToSymbol(alpha, &h_alpha, sizeof(int)));
		CUCall(cudaMemcpyToSymbol(m, &h_m, sizeof(int)));
		CUCall(cudaMemcpyToSymbol(a, &h_a, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(kk, &h_kk, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(dt, &h_dt, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(epsilon, &h_epsilon, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(M1, &h_M1, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(M2, &h_M2, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(b, &h_b, sizeof(double)));

		size_t size = (n + 2) * (m + 2) * sizeof(double);

		//init host memory
		h_E = (double*)malloc(size);
		h_E_prev = (double*)malloc(size);
		h_R = (double*)malloc(size);

		CUCall(cudaMalloc((void**)&d_E, size));
		CUCall(cudaMalloc((void**)&d_E_prev, size));
		CUCall(cudaMalloc((void**)&d_R, size));

		//Convert 2d arrays to 1D
		size_t n_row = m + 2, n_col = n + 2;
		size_t idx = 0;
		for(size_t i = 0; i < n_row; i++){
			for(size_t j = 0; j < n_col; j++){
				idx = (i * n_col) + j;
				h_E[idx] = E[i][j];
				h_E_prev[idx] = E_prev[i][j];
				h_R[idx] = R[i][j];
			}
		}

		CUCall(cudaMemcpy(d_E, h_E, size, cudaMemcpyHostToDevice));
		CUCall(cudaMemcpy(d_E_prev, h_E_prev, size, cudaMemcpyHostToDevice));
		CUCall(cudaMemcpy(d_R, h_R, size, cudaMemcpyHostToDevice));
	}

	void call(void){
		dim3 threadsPerBlock(16, 1);
		dim3 numBlocks(1, 1);
		simulate_v1_PDE<<<numBlocks, threadsPerBlock>>>(d_E, d_E_prev, d_R);
		simulate_v1_ODE<<<numBlocks, threadsPerBlock>>>(d_E, d_E_prev, d_R);
	}

	

	void end(void){
		free(h_E);
		free(h_E_prev);
		free(h_R);

		CUCall(cudaFree(d_E));
		CUCall(cudaFree(d_E_prev));
		CUCall(cudaFree(d_R));
	}

    void load(double ***E, double ***R, double ***E_prev){
		size_t size = (n + 2) * (m + 2) * sizeof(double);
		CUCall(cudaMemcpy(h_E, d_E, size, cudaMemcpyDeviceToHost));
		CUCall(cudaMemcpy(h_E_prev, d_E_prev, size, cudaMemcpyDeviceToHost));
		CUCall(cudaMemcpy(h_R, d_R, size, cudaMemcpyDeviceToHost));
		CUCall(cudaDeviceSynchronize());

		size_t n_row = m + 2, n_col = n + 2;
		size_t idx = 0;
		for(size_t i = 0; i < n_row; i++){
			for(size_t j = 0; j < n_col; j++){
				idx = (i * n_col) + j;
				(*E)[i][j] = h_E[idx];
				(*E_prev)[i][j] = h_E_prev[idx];
				(*R)[i][j] = h_R[idx];
			}
		}
		


	}
	
}
