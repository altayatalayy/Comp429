/*

	Implement your CUDA kernel in this file

*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#include "cardiacsim_kernels.cuh"

//#define __constant__
//#define __global__

__constant__ int n;
__constant__ int m;
__constant__ double alpha;
__constant__ double kk;
__constant__ double dt;
__constant__ double epsilon;
__constant__ double M1;
__constant__ double M2;
__constant__ double b;
__constant__ double a;


void check2(cudaError_t err, const char * file, int line){
    if(err != cudaSuccess){
        printf("Cuda error occured, %s at line %d\nName:%s\nDescription:%s\n",\
            file, line, cudaGetErrorName(err), cudaGetErrorString(err));
    }
}
#undef __DEBUG__
#define __DEBUG__

#ifdef __DEBUG__
#define CUCall(x) (x); check2((cudaError_t)cudaGetLastError(), __FILE__, __LINE__)
#else
#define CUCall(x) (x)
#endif

__global__ void print_consts(){
	printf("n = %d, m = %d, alpha = %f, kk = %f\n", n, m, alpha, kk);
	printf("dt = %f, epsiloen = %f, M1 = %f, M2 = %f, b = %f, a = %f\n", dt, epsilon, M1, M2, b, a);
}

__global__ void sim_boundry1(double *E_prev){
	int t_row_idx = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
	size_t r_index;
	if ((t_row_idx > 0) && (t_row_idx < m+2 )){
		r_index = (size_t)(t_row_idx * (m + 2));
		E_prev[r_index + 0] = E_prev[r_index + 2];
		E_prev[r_index + (size_t)n+1] = E_prev[r_index + (size_t)n-1];
	}
}


__global__ void sim_boundry2(double *E_prev){
	int t_col_idx = (blockIdx.x * blockDim.x) + threadIdx.x + 1; //thread row and column indeces.

	if ((t_col_idx > 0) && (t_col_idx < n+2)){
		E_prev[0 * (m+2) + t_col_idx] = E_prev[2 * (m+2) + t_col_idx];
		E_prev[(m+1) * (m+2) + t_col_idx] = E_prev[(m-1) * (m+2) + t_col_idx];
	}
}



__global__ void simulate_v1_PDE(double *E, double *R, double *E_prev){
	int t_row_idx = (blockIdx.x * blockDim.x) + threadIdx.x + 1; //thread row and column indeces.
	int t_col_idx = (blockIdx.y * blockDim.y) + threadIdx.y + 1;
	//printf("row = %d, col = %d\n", row_idx, col_idx);
	//printf("by = %d: %d * %d + %d\n", t_col_idx, blockIdx.y, blockDim.y, threadIdx.y);
	//printf("n = %d, m = %d, t_col = %d, t_row = %d\n", n, m, t_col_idx, t_row_idx);

	if ((t_row_idx > 0) && (t_col_idx > 0) && (t_row_idx < m+2 ) && (t_col_idx < n+2)){
		size_t index = (size_t)(t_row_idx * (m + 2) + t_col_idx);
		size_t left = index - 1, right = index + 1, up = index - (m+2), down = index + (m+2);
		//printf("E_prev[%2d][%2d] = %4.2f, index = %u\n", t_row_idx, t_col_idx, E_prev[index], (uint)index);
		E[index] = E_prev[index] + alpha * (E_prev[up] + E_prev[down] - 4*E_prev[index] + E_prev[right] + E_prev[left]); 
	}
}
__global__ void simulate_v1_ODE(double *E, double *R, double *E_prev){
	int t_row_idx = blockIdx.x*blockDim.x + threadIdx.x + 1; //thread row and column indeces.
	int t_col_idx = blockIdx.y*blockDim.y + threadIdx.y + 1;

	if((t_row_idx > 0) && (t_col_idx > 0) && (t_row_idx < n+2 ) && (t_col_idx < n+2)){
		size_t index = (size_t)(t_row_idx * (m+2) + t_col_idx);
		E[index] = E[index] - dt * (kk * E[index] * (E[index] - a) * (E[index] - 1) + E[index] * R[index]);
		R[index] = R[index] + dt * (epsilon + M1 * R[index] / (E[index] + M2)) * (-R[index] - kk * E[index] * (E[index] - b - 1));
	}
    
}    
__global__ void simulate_v2(double *E, double *R, double *E_prev){
	int t_row_idx = blockIdx.x*blockDim.x + threadIdx.x + 1; //thread row and column indeces.
	int t_col_idx = blockIdx.y*blockDim.y + threadIdx.y + 1;

	if((t_row_idx > 0) && (t_col_idx > 0) && (t_row_idx < n+2 ) && (t_col_idx < n+2)){
		size_t index = (size_t)(t_row_idx * (m + 2) + t_col_idx);
		size_t left = index - 1, right = index + 1, up = index - (m+2), down = index + (m+2);
		E[index] = E_prev[index] + alpha * (E_prev[up] + E_prev[down] - 4*E_prev[index] + E_prev[right] + E_prev[left]); 
		E[index] = E[index] - dt * (kk * E[index] * (E[index] - a) * (E[index] - 1) + E[index] * R[index]);
		R[index] = R[index] + dt * (epsilon + M1 * R[index] / (E[index] + M2)) * (-R[index] - kk * E[index] * (E[index] - b - 1));
	}
    
}

__global__ void simulate_v3(double *E, double *R, double *E_prev){
	int t_row_idx = blockIdx.x*blockDim.x + threadIdx.x + 1; //thread row and column indeces.
	int t_col_idx = blockIdx.y*blockDim.y + threadIdx.y + 1;

	if((t_row_idx > 0) && (t_col_idx > 0) && (t_row_idx < n+2 ) && (t_col_idx < n+2)){
		size_t index = (size_t)(t_row_idx * (m + 2) + t_col_idx);
		size_t left = index - 1, right = index + 1, up = index - (m+2), down = index + (m+2);
		double e, r = R[index];
		e = E_prev[index] + alpha * (E_prev[up] + E_prev[down] - 4*E_prev[index] + E_prev[right] + E_prev[left]); 
		e = e - dt * (kk * e * (e - a) * (e - 1) + e * r);
		R[index] = r + dt * (epsilon + M1 * r / (e + M2)) * (-r - kk * e * (e - b - 1));
		E[index] = e;
	}
}


#define BN 16
#define BM 16
__global__ void simulate_v4(double *E, double *R, double *E_prev){
	const int bn = BN, bm = BM;
	__shared__ double lE[(bn + 2) * (bm + 2)];
	int idx = (threadIdx.y + 1) * (blockDim.x + 2) + (threadIdx.x + 1);
	int lidx = idx;
	size_t gidx = threadIdx.x + 1 + blockIdx.x * blockDim.x + (blockIdx.y * blockDim.y + threadIdx.y + 1) * (n + 2);//idx + blockIdx.x * blockDim.x + blockIdx.y * (n + 2);//(size_t)(blockIdx.x + 1) * blockDim.x + (n + 2) * (threadIdx.y + 1 + blockIdx.y * bn) - threadIdx.x;
	lE[idx] = E_prev[gidx];
	//printf("gidx = %d, idx = %d\n", gidx, idx);
	//E[gidx] = -1;//lE[idx];
	//load host cells
	if(threadIdx.y == 0){
		idx = threadIdx.x + 1;
		gidx = (size_t)(n + 2) * (blockIdx.y * blockDim.y) + blockIdx.x * blockDim.x + idx;
		//printf("gidx = %d, idx = %d\n", gidx, idx);
		lE[idx] = E_prev[gidx];
		//E[gidx] = -1;//lE[idx];
	}

	if(threadIdx.y == bm-1){
		idx = threadIdx.x + 1 + (bm+1) * (bn+2);
		gidx = (size_t)(n + 2) * (blockIdx.y * blockDim.y + (bm + 1)) + blockIdx.x * blockDim.x + threadIdx.x + 1;
		lE[idx] = E_prev[gidx];
		//printf("gidx = %d, idx = %d\n", gidx, idx);
		//E[gidx] = -1;//lE[idx];
	}

	if(threadIdx.x == 0){
		idx = (threadIdx.y + 1) * (blockDim.x + 2);
		gidx = (size_t)(n + 2) * (threadIdx.y + 1 + blockIdx.y * blockDim.y) + (blockIdx.x * (blockDim.x + 0));
		lE[idx] = E_prev[gidx];
		//printf("gidx = %d, idx = %d\n", gidx, idx);
		//E[gidx] = -1;//lE[idx];
	}

	if(threadIdx.x == bn-1){
		idx = (threadIdx.y + 1) * (blockDim.x + 2) + (bn + 1);
		gidx = (size_t)(bn + 1) + (n + 2) * (threadIdx.y + 1 + blockIdx.y * blockDim.y) + (blockIdx.x * (blockDim.x + 0));
		lE[idx] = E_prev[gidx];
		//printf("gidx = %d, idx = %d\n", gidx, idx);
		//E[gidx] = -1;//lE[idx];
	}	

	//int idx = (threadIdx.y + 1) * (blockDim.x + 2) + (threadIdx.x + 1);
	//gidx = (size_t)(blockIdx.x + 1) * blockDim.x + (n + 2) * (threadIdx.y + 1 + blockIdx.y * bn) - threadIdx.x;
	//int lidx = (threadIdx.y + 1) * (blockDim.x + 2) + (threadIdx.x + 1);
	size_t left = lidx - 1, right = lidx + 1, up = lidx - (blockDim.x+2), down = lidx + (blockDim.x+2);
	
	int t_row_idx = blockIdx.x*blockDim.x + threadIdx.x + 1; //thread row and column indeces.
	int t_col_idx = blockIdx.y*blockDim.y + threadIdx.y + 1;
	//printf("r = %d, c = %d", t_row_idx, t_col_idx);

	//size_t index = (size_t)(t_row_idx * (m + 2) + t_col_idx);
	//size_t index = (size_t)(blockIdx.x + 1) * blockDim.x + (n + 2) * (threadIdx.y + 1 + blockIdx.y * bn) - threadIdx.x;
	size_t index = threadIdx.x + 1 + blockIdx.x * blockDim.x + (blockIdx.y * blockDim.y + threadIdx.y + 1) * (n + 2);
	double e, r = R[index];

	__syncthreads();
	
	e = lE[lidx] + alpha * (lE[up] + lE[down] - 4*lE[lidx] + lE[right] + lE[left]); 
	e = e - dt * (kk * e * (e - a) * (e - 1) + e * r);
	R[index] = r + dt * (epsilon + M1 * r / (e + M2)) * (-r - kk * e * (e - b - 1));
	E[index] = e;
}

namespace Simulation{
	double *h_E, *h_E_prev, *h_R;
	double *d_E, *d_E_prev, *d_R;

	int hn, hm;

	void init(double **E, double **E_prev, double **R, double h_alpha, int h_n, int h_m, double h_a,double h_kk, double h_dt, double h_epsilon, double h_M1, double h_M2, double h_b){
		//printf("%d\n", h_m);
		hn = h_n; hm = h_m;
		//init consts
		CUCall(cudaMemcpyToSymbol(alpha, &h_alpha, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(n, &h_n, sizeof(int)));
		CUCall(cudaMemcpyToSymbol(m, &h_m, sizeof(int)));
		CUCall(cudaMemcpyToSymbol(a, &h_a, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(kk, &h_kk, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(dt, &h_dt, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(epsilon, &h_epsilon, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(M1, &h_M1, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(M2, &h_M2, sizeof(double)));
		CUCall(cudaMemcpyToSymbol(b, &h_b, sizeof(double)));

		size_t size = (hn + 2) * (hm + 2) * sizeof(double);

		//init host memory
		h_E = (double*)malloc(size);
		h_E_prev = (double*)malloc(size);
		h_R = (double*)malloc(size);

		CUCall(cudaMalloc((void**)&d_E, size));
		CUCall(cudaMalloc((void**)&d_E_prev, size));
		CUCall(cudaMalloc((void**)&d_R, size));

		//Convert 2d arrays to 1D
		size_t n_row = hm + 2, n_col = hn + 2;
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
		//print_consts<<<1, 1>>>();
	}

	void call(void){
		static const int tx = BN, ty = BM;
		static const int bx = (hn % tx == 0) ? (hn / tx) : (hn / tx) + 1;
		static const int by = (hn % ty == 0) ? (hn / ty) : (hn / ty) + 1;
		//printf("bx = %d, by = %d\n", bx, by);
		static const dim3 threadsPerBlock(tx, ty);
		static const dim3 numBlocks(bx, by);

		sim_boundry1<<<dim3(1, by), dim3(1, ty)>>>(d_E_prev);
		sim_boundry2<<<dim3(bx, 1), dim3(tx, 1)>>>(d_E_prev);
		//simulate_v1_PDE<<<numBlocks, threadsPerBlock>>>(d_E, d_R, d_E_prev);
		//simulate_v1_ODE<<<numBlocks, threadsPerBlock>>>(d_E, d_R, d_E_prev);
		//simulate_v2<<<numBlocks, threadsPerBlock>>>(d_E, d_R, d_E_prev);
		simulate_v3<<<numBlocks, threadsPerBlock>>>(d_E, d_R, d_E_prev);
		//simulate_v4<<<numBlocks, threadsPerBlock>>>(d_E, d_R, d_E_prev);
		CUCall(cudaDeviceSynchronize());
		double *tmp = d_E; d_E = d_E_prev; d_E_prev = tmp;
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
		size_t size = (hn + 2) * (hm + 2) * sizeof(double);
		CUCall(cudaMemcpy(h_E, d_E, size, cudaMemcpyDeviceToHost));
		CUCall(cudaMemcpy(h_E_prev, d_E_prev, size, cudaMemcpyDeviceToHost));
		CUCall(cudaMemcpy(h_R, d_R, size, cudaMemcpyDeviceToHost));
		CUCall(cudaDeviceSynchronize());

		size_t n_row = hm + 2, n_col = hn + 2;
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
