#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include <cuda_runtime.h> 

#ifndef MINF
#ifdef __CUDACC__
#define MINF __int_as_float(0xff800000)
#else
#define  MINF (std::numeric_limits<float>::infinity())
#endif
#endif 

//#define Stereo_ENABLED
#define LE_THREAD_SIZE 16

struct SolverInput
{
	unsigned int N;					// Number of vertices
	unsigned int M;					// Number of variables

	float wReg;

	int* d_numNeighbours;
	int* d_neighbourIdx;
	int* d_neighbourOffset;
};

struct SolverState
{
	// State of the GN Solver
	float2*  d_target;		
	float3*	 d_mesh;
	float3*  d_average;
	float*	 d_blendshapeWeights;
	float3*	 d_blendshapeBasis;
    float*   d_camParams;

	float2* d_Jv;
	float*  d_JTv;
	float*  d_Jv2;

	float*	 d_delta;

	float*	d_r;

	float*	d_z;

	float*	d_p;		
	
	float*	d_Ap;
	
	float*	d_scanAlpha;				
	float*	d_scanBeta;					
	float*	d_rDotzOld;					
	
	float*	d_precondioner;

	float*	d_sumResidual;			

	__host__ float getSumResidual() const {
		float residual;
		cudaMemcpy(&residual, d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
		return residual;
	}
};

#endif
