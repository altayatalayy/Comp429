#include <cuda.h>
#include <cuda_runtime.h>

#pragma once

#define N 256
#define M 256

namespace Simulation{
	void init(double **E, double **E_prev, double **R, double h_alpha, int h_n, int h_m,double h_a, double h_kk, double h_dt, double h_epsilon, double h_M1, double h_M2, double h_b);
    void call(void);
    void end(void);
    void load(double ***E, double ***R, double ***E_prev);
    
}