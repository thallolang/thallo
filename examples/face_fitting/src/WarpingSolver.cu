#include <iostream>

// Enabled to print a bunch of junk during solving
#define DEBUG_PRINT_SOLVER_INFO 0

#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"
#include "WarpingSolverUtil.h"
#include "WarpingSolverEquations.h"

#include <assert.h>
#include <stdio.h>
#include <stdint.h>

#include "../../shared/CUDATimer.h"

#ifdef _WIN32
#include <conio.h>
#endif

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#define WARP_SIZE 32u
#define WARP_MASK (WARP_SIZE-1u)

/////////////////////////////////////////////////////////////////////////
// Eval Residual
/////////////////////////////////////////////////////////////////////////

__global__ void ResetResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0) state.d_sumResidual[0] = 0.0f;
}

__global__ void EvalResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of residuals
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float residual = 0.0f;
	if (x < N)
	{
		residual = evalFDevice(x, input, state, parameters);
	}
	residual = warpReduce(residual);

	unsigned int laneid;
	//This command gets the lane ID within the current warp
	asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
	if (laneid == 0) {
	  atomicAdd(&state.d_sumResidual[0], residual);
	}
}

float EvalResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	float residual = 0.0f;

	const unsigned int N = input.N; // Number of vertices
	ResetResidualDevice << < 1, 1, 1 >> >(input, state, parameters);
	cudaSafeCall(cudaDeviceSynchronize());
	timer.startEvent("EvalResidual");
	EvalResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
	timer.endEvent();
	cudaSafeCall(cudaDeviceSynchronize());

	residual = state.getSumResidual();

	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif

	return residual;
}

/////////////////////////////////////////////////////////////////////////
// Compute JTF Device
/////////////////////////////////////////////////////////////////////////


__global__ void computeJTFDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of residuals
	const unsigned int M = input.M; // Number of variables

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x < N*M)
	{
		unsigned int i = x % N;
		unsigned int j = x / N;

		float3 v = state.d_mesh[i];
		float2 t = state.d_target[i];

		float res = 0.0f;
		if (t.x != MINF)
		{
            float3 bb = state.d_blendshapeBasis[j*N + i];

            float2 JT = snavelyDerivatives(state.d_camParams, v, bb);
            float2 F = t-snavelyProjection(state.d_camParams, v);

			res += dot(JT, F);
		}
		if (i == 0)
		{
			float F = state.d_blendshapeWeights[j];
			float JT = 1.0f;
			res += input.wReg*JT*F;
		}

		atomicAdd(&state.d_JTv[j], res);
	}
}

void ComputeJTF(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	const unsigned int N = input.N;
	const unsigned int M = input.M;

	cudaSafeCall(cudaMemset(state.d_JTv, 0, M*sizeof(float)));

	const int blocksPerGrid = (N*M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	computeJTFDevice << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);

	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif
}

/////////////////////////////////////////////////////////////////////////
// Compute JTJp Device
/////////////////////////////////////////////////////////////////////////

__global__ void computeJTJpDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of residuals
	const unsigned int M = input.M; // Number of variables

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N*M)
	{
		unsigned int i = x % N;
		unsigned int j = x / N;

		float2 t = state.d_target[i];
		float res = 0.0f;
		if (t.x != MINF)
		{
			float2 Jp = state.d_Jv[i];

	        float3 bb = state.d_blendshapeBasis[j*N + i];

	        float3 pt = state.d_mesh[i];
	        float2 JT = snavelyDerivatives(state.d_camParams, pt, bb);
	        res += dot(JT, Jp);
		}


		if (i == 0)
		{
			float JT = 1.0f;
			res += sqrt(input.wReg)*JT*state.d_Jv2[j];
		}

		atomicAdd(&state.d_JTv[j], res);
	}
}


void ComputeJTJp(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	const unsigned int N = input.N;
	const unsigned int M = input.M;

	cudaSafeCall(cudaMemset(state.d_JTv, 0, M*sizeof(float)));

	const int blocksPerGrid = (N*M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	computeJTJpDevice << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);

	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif
}

/////////////////////////////////////////////////////////////////////////
// Compute Jp Device
/////////////////////////////////////////////////////////////////////////

__global__ void computeJpDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of residuals
	const unsigned int M = input.M; // Number of variables

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N*M)
	{
		unsigned int i = x % N;
		unsigned int j = x / N;

		float2 t = state.d_target[i];
		if (t.x != MINF)
		{
	        float3 pt = state.d_mesh[i];
	        float3 bb = state.d_blendshapeBasis[j*N + i];
	        float2 Jp = snavelyDerivatives(state.d_camParams, pt, bb) * state.d_p[j];
			atomicAdd(&state.d_Jv[i].x, Jp.x);
			atomicAdd(&state.d_Jv[i].y, Jp.y);
		}

		if (i == 0)
		{
			float v = sqrt(input.wReg)*state.d_p[j];
			atomicAdd(&state.d_Jv2[j], v);
		}
	}
}

void ComputeJp(SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	const unsigned int N = input.N;
	const unsigned int M = input.M;

	cudaSafeCall(cudaMemset(state.d_Jv, 0, N*sizeof(float2)));
	cudaSafeCall(cudaMemset(state.d_Jv2, 0, M*sizeof(float)));

	const int blocksPerGrid = (N*M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	computeJpDevice << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);

	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif
}

// For the naming scheme of the variables see:
// http://en.wikipedia.org/wiki/Conjugate_gradient_method
// This code is an implementation of their PCG pseudo code

__global__ void PCGInit_Kernel0(SolverInput input, SolverState state)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < input.M) {
		state.d_delta[x] = 0.0f;
		state.d_r[x] = 0.0f;
		state.d_p[x] = 0.0f;
		state.d_z[x] = 0.0f;
		state.d_Ap[x] = 0.0f;
		state.d_precondioner[x] = 1.0f;
	}
}

__global__ void PCGInit_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int M = input.M;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x < M)
	{
		const float residuum = -state.d_JTv[x]; // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 

		state.d_r[x]  = residuum;												 // store for next iteration

		const float p  = state.d_precondioner[x]  * residuum;					 // apply preconditioner M^-1
		state.d_p[x] = p;

		d = residuum*p;															 // x-th term of nomimator for computing alpha and denominator for computing beta
	}
	
	d = warpReduce(d);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_scanAlpha, d);
    }
}
 
__global__ void PCGInit_Kernel2(SolverInput input, SolverState state)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < input.M) {
        state.d_rDotzOld[x] = state.d_scanAlpha[0];
    }
}

void Initialization(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	const unsigned int M = input.M;

	const int blocksPerGrid = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int shmem_size = sizeof(float)*THREADS_PER_BLOCK;

	cudaSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float)));
    
	//timer.startEvent("PCGInit_Kernel0");
	PCGInit_Kernel0 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state);
	//timer.endEvent();
	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif

	ComputeJTF(input, state, parameters);
	
	//timer.startEvent("PCGInit_Kernel1");
	PCGInit_Kernel1 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state, parameters);
    //timer.endEvent();
	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif

	//timer.startEvent("PCGInit_Kernel2");
	PCGInit_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state);
	//timer.endEvent();
	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif

	#if DEBUG_PRINT_SOLVER_INFO 
	    float temp;
	    cudaSafeCall(        cudaMemcpy(&temp, state.d_scanAlpha, sizeof(float), cudaMemcpyDeviceToHost) );
	    printf("ScanAlpha (Init): %f\n", temp);
	#endif
}

/////////////////////////////////////////////////////////////////////////
// PCG Iteration Parts
/////////////////////////////////////////////////////////////////////////

__global__ void PCGStep_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int M = input.M;											// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x < M)
	{
		const float tmp = state.d_JTv[x];								// A x p_k  => J^T x J x p_k 
		state.d_Ap[x] = tmp;											// store for next kernel call

		d = state.d_p[x]*tmp;											// x-th term of denominator of alpha
	}

	d = warpReduce(d);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_scanAlpha, d); // sum over x-th terms to compute denominator of alpha inside this block
    }
}

__global__ void PCGStep_Kernel2(SolverInput input, SolverState state)
{
	const unsigned int M = input.M;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	const float dotProduct = state.d_scanAlpha[0];

	float b = 0.0f;
	if (x < M)
	{
		float alpha = 0.0f;
		if (dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x] / dotProduct;   // update step size alpha

		state.d_delta[x]  = state.d_delta[x]  + alpha*state.d_p[x];					// do a decent step
	
		float r  = state.d_r[x] - alpha*state.d_Ap[x];					// update residuum
		state.d_r[x] = r;												// store for next kernel call

		float z  = state.d_precondioner[x] * r;							// apply preconditioner M^-1
		state.d_z[x] = z;												// save for next kernel call

		b = z*r;														// compute x-th term of the nominator of beta
	}

	b = warpReduce(b);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_scanBeta, b); // sum over x-th terms to compute denominator of alpha inside this block
    }
}

__global__ void PCGStep_Kernel3(SolverInput input, SolverState state)
{
	const unsigned int M = input.M;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < M)
	{
		const float rDotzNew = state.d_scanBeta[0];									// get new nominator
		const float rDotzOld = state.d_rDotzOld[x];									// get old denominator

		float beta = 0.0f;
		if (rDotzOld > FLOAT_EPSILON) beta = rDotzNew / rDotzOld;					// update step size beta

		state.d_rDotzOld[x] = rDotzNew;												// save new rDotz for next iteration
		state.d_p[x]  = state.d_z[x]  + beta*state.d_p[x];							// update decent direction
	}
}

void PCGIteration(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	const unsigned int M = input.M;	// Number of block variables

	ComputeJp(input, state, parameters);
	ComputeJTJp(input, state, parameters);

	// Do PCG step
	const int blocksPerGrid = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int shmem_size = sizeof(float)*THREADS_PER_BLOCK;

	cudaSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float)));
    //timer.startEvent("PCGStep_Kernel1");
	PCGStep_Kernel1 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state, parameters);
    //timer.endEvent();
	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif

	cudaSafeCall(cudaMemset(state.d_scanBeta, 0, sizeof(float)));
	//timer.startEvent("PCGStep_Kernel2");
	PCGStep_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state);
	//timer.endEvent();
	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif

	//timer.startEvent("PCGStep_Kernel3");
	PCGStep_Kernel3 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state);
	//timer.endEvent();
	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif
	#if DEBUG_PRINT_SOLVER_INFO 
	    float temp;
	    cudaSafeCall( cudaMemcpy(&temp, state.d_scanAlpha, sizeof(float), cudaMemcpyDeviceToHost) );
	    printf("ScanAlpha (Step): %f\n", temp);
	    cudaSafeCall( cudaMemcpy(&temp, state.d_scanBeta, sizeof(float), cudaMemcpyDeviceToHost) );
	    printf("ScanBeta (Step): %f\n", temp);
    #endif
}

/////////////////////////////////////////////////////////////////////////
// Apply Update
/////////////////////////////////////////////////////////////////////////

__global__ void ApplyLinearUpdateDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < input.M) {
		state.d_blendshapeWeights[x] = state.d_blendshapeWeights[x] + state.d_delta[x];
	}
}

void ApplyLinearUpdate(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	const unsigned int M = input.M; // This is different to all sparse solvers !!!
	//timer.startEvent("ApplyLinearUpdateDevice");
	ApplyLinearUpdateDevice << <(M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
    //timer.endEvent();
	cudaSafeCall(cudaDeviceSynchronize()); // Hm

	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif
}

/////////////////////////////////////////////////////////////////////////
// Transform Mesh
/////////////////////////////////////////////////////////////////////////

__global__ void MorphMeshDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // number of vertices
	const unsigned int M = input.M; // number of variables

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (x < N*M)
	{
		unsigned int i = x % N;
		unsigned int j = x / N;
		
		float3 a = make_float3(0.0f, 0.0f, 0.0f);	
		if (j == 0)
		{
			a += state.d_average[i];
		}

		float3 d = a + state.d_blendshapeBasis[j*input.N + i] * state.d_blendshapeWeights[j];

		atomicAdd(&state.d_mesh[i].x, d.x);
		atomicAdd(&state.d_mesh[i].y, d.y);
		atomicAdd(&state.d_mesh[i].z, d.z);
	}
}

void MorphMesh(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	const unsigned int N = input.N; // number of vertices
	const unsigned int M = input.M; // number of variables
	timer.startEvent("Transform Mesh");

	cudaSafeCall(cudaMemset(state.d_mesh, 0, N*sizeof(float3)));
	MorphMeshDevice << <(N*M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
	timer.endEvent();
	cudaSafeCall(cudaDeviceSynchronize()); // Hm

	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif
}

////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

extern "C" double ImageWarpingSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverPerformanceSummary& stats)
{
    CUDATimer timer;
    timer.startEvent("Total");
	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
        timer.startEvent("Nonlinear Iteration");
        timer.startEvent("Nonlinear Setup");
		MorphMesh(input, state, parameters, timer);
		float residual = EvalResidual(input, state, parameters, timer);
		printf("%i: cost: %f\n", nIter, residual);
	
		Initialization(input, state, parameters, timer);
        timer.endEvent();
	
        timer.startEvent("Linear Solve");
		for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++) {
			MorphMesh(input, state, parameters, timer);  // not needed here, since this problem is linear so far
			PCGIteration(input, state, parameters, timer);
		}
        timer.endEvent();
	
        timer.startEvent("Nonlinear Finish");
		ApplyLinearUpdate(input, state, parameters, timer);
        timer.endEvent();
	
        timer.nextIteration();
        timer.endEvent();
	}

	MorphMesh(input, state, parameters, timer);
	float residual = EvalResidual(input, state, parameters, timer);
	printf("final cost: %f\n", residual);
    timer.endEvent();
    timer.evaluate(stats);
    return (double)residual;
}
