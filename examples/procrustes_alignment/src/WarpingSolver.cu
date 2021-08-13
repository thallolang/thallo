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

	const unsigned int N = input.N; // Number of residuals
	ResetResidualDevice << < 1, 1, 1 >> >(input, state, parameters);
	cudaSafeCall(cudaDeviceSynchronize());
	//timer.startEvent("EvalResidual");
	EvalResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
	//timer.endEvent();
	cudaSafeCall(cudaDeviceSynchronize());

	residual = state.getSumResidual();

	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif

	return residual;
}

// For the naming scheme of the variables see:
// http://en.wikipedia.org/wiki/Conjugate_gradient_method
// This code is an implementation of their PCG pseudo code

__global__ void PCGInit_Kernel0(unsigned int N, SolverState state)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
		state.d_delta[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_deltaA[x] = make_float3(0.0f, 0.0f, 0.0f);

		state.d_r[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_rA[x] = make_float3(0.0f, 0.0f, 0.0f);

		state.d_p[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_pA[x] = make_float3(0.0f, 0.0f, 0.0f);

		state.d_z[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_zA[x] = make_float3(0.0f, 0.0f, 0.0f);

		state.d_Ap_X[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_Ap_A[x] = make_float3(0.0f, 0.0f, 0.0f);

		state.d_precondioner[x] = make_float3(1.0f, 1.0f, 1.0f);
		state.d_precondionerA[x] = make_float3(1.0f, 1.0f, 1.0f);
	}
}

__global__ void PCGInit_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x < N)
	{
		float3 residuumA;
		const float3 residuum = evalMinusJTFDevice(x, input, state, parameters, residuumA); // residuum = J^T x -F - A x delta_0  => J^T x -F, since A x x_0 == 0 

		//state.d_r[x]  = residuum;												 // store for next iteration
		//state.d_rA[x] = residuumA;												 // store for next iteration

		atomicAdd(&state.d_r[0].x, residuum.x);
		atomicAdd(&state.d_r[0].y, residuum.y);
		atomicAdd(&state.d_r[0].z, residuum.z);

		atomicAdd(&state.d_rA[0].x, residuumA.x);
		atomicAdd(&state.d_rA[0].y, residuumA.y);
		atomicAdd(&state.d_rA[0].z, residuumA.z);

		const float3 p  = state.d_precondioner[0]  * residuum;					 // apply preconditioner M^-1
		//state.d_p[x] = p;

		const float3 pA = state.d_precondionerA[0] * residuumA;					 // apply preconditioner M^-1
		//state.d_pA[x] = pA;

		atomicAdd(&state.d_p[0].x, p.x);
		atomicAdd(&state.d_p[0].y, p.y);
		atomicAdd(&state.d_p[0].z, p.z);

		atomicAdd(&state.d_pA[0].x, pA.x);
		atomicAdd(&state.d_pA[0].y, pA.y);
		atomicAdd(&state.d_pA[0].z, pA.z);

		d = dot(residuum, p) + dot(residuumA, pA);								 // x-th term of nomimator for computing alpha and denominator for computing beta
	}
	
	d = warpReduce(d);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_scanAlpha, d);
    }
}
 
__global__ void PCGInit_Kernel2(unsigned int N, SolverState state)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N) {
        state.d_rDotzOld[x] = state.d_scanAlpha[0];
        //state.d_delta[x] = make_float3(0.0f, 0.0f, 0.0f);
        //state.d_deltaA[x] = make_float3(0.0f, 0.0f, 0.0f);
		//
		//state.d_r[x] = make_float3(0.0f, 0.0f, 0.0f);
		//state.d_rA[x] = make_float3(0.0f, 0.0f, 0.0f);
		//
		//state.d_p[x] = make_float3(0.0f, 0.0f, 0.0f);
		//state.d_pA[x] = make_float3(0.0f, 0.0f, 0.0f);
		//
		////state.d_z[x] = make_float3(0.0f, 0.0f, 0.0f);
		////state.d_zA[x] = make_float3(0.0f, 0.0f, 0.0f);
		//
		//state.d_Ap_X[x] = make_float3(0.0f, 0.0f, 0.0f);
		//state.d_Ap_A[x] = make_float3(0.0f, 0.0f, 0.0f);
    }
}

void Initialization(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	const unsigned int N = input.N;

	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int shmem_size = sizeof(float)*THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}
	cudaSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float)));
    
	//timer.startEvent("PCGInit_Kernel0");
	PCGInit_Kernel0 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(1, state);
	//timer.endEvent();
	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif
	
	//timer.startEvent("PCGInit_Kernel1");
	PCGInit_Kernel1 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state, parameters);
    //timer.endEvent();
	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif

	//timer.startEvent("PCGInit_Kernel2");
	PCGInit_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(1, state);
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
	const unsigned int N = input.N;											// Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x < N)
	{
		float3 tmpA;
		const float3 tmp = applyJTJDevice(x, input, state, parameters, tmpA);		// A x p_k  => J^T x J x p_k 

		//if (x < 1)
		{
			atomicAdd(&state.d_Ap_X[0].x, tmp.x);
			atomicAdd(&state.d_Ap_X[0].y, tmp.y);
			atomicAdd(&state.d_Ap_X[0].z, tmp.z);

			atomicAdd(&state.d_Ap_A[0].x, tmpA.x);
			atomicAdd(&state.d_Ap_A[0].y, tmpA.y);
			atomicAdd(&state.d_Ap_A[0].z, tmpA.z);

			//state.d_Ap_X[x] = tmp;																	// store for next kernel call
			//state.d_Ap_A[x] = tmpA;																// store for next kernel call
		}

		d = dot(state.d_p[0], tmp) + dot(state.d_pA[0], tmpA);									// x-th term of denominator of alpha
	}

	d = warpReduce(d);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_scanAlpha, d); // sum over x-th terms to compute denominator of alpha inside this block
    }
}

__global__ void PCGStep_Kernel2(SolverInput input, SolverState state)
{
	const unsigned int N = 1;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	const float dotProduct = state.d_scanAlpha[0];

	float b = 0.0f;
	if (x < N)
	{
		float alpha = 0.0f;
		if (dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x] / dotProduct;   // update step size alpha

		state.d_delta[x]  = state.d_delta[x]  + alpha*state.d_p[x];					// do a decent step
		state.d_deltaA[x] = state.d_deltaA[x] + alpha*state.d_pA[x];				// do a decent step

		float3 r  = state.d_r[x] - alpha*state.d_Ap_X[x];					// update residuum
		state.d_r[x] = r;													// store for next kernel call

		float3 rA = state.d_rA[x] - alpha*state.d_Ap_A[x];					// update residuum
		state.d_rA[x] = rA;													// store for next kernel call

		float3 z  = state.d_precondioner[x] * r;							// apply preconditioner M^-1
		state.d_z[x] = z;													// save for next kernel call

		float3 zA = state.d_precondionerA[x] * rA;							// apply preconditioner M^-1
		state.d_zA[x] = zA;													// save for next kernel call

		b = dot(z, r) + dot(zA, rA);										// compute x-th term of the nominator of beta
	}

	b = warpReduce(b);
    if ((threadIdx.x & WARP_MASK) == 0) {
        atomicAdd(state.d_scanBeta, b); // sum over x-th terms to compute denominator of alpha inside this block
    }
}

__global__ void PCGStep_Kernel3(SolverInput input, SolverState state)
{
	const unsigned int N = 1;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N)
	{
		const float rDotzNew = state.d_scanBeta[0];									// get new nominator
		const float rDotzOld = state.d_rDotzOld[x];									// get old denominator

		float beta = 0.0f;
		if (rDotzOld > FLOAT_EPSILON) beta = rDotzNew / rDotzOld;					// update step size beta

		state.d_rDotzOld[x] = rDotzNew;												// save new rDotz for next iteration
		state.d_p[x]  = state.d_z[x]  + beta*state.d_p[x];							// update decent direction
		state.d_pA[x] = state.d_zA[x] + beta*state.d_pA[x];							// update decent direction
	}
}

void PCGIteration(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	const unsigned int N = input.N;	// Number of block variables

	// Do PCG step
	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int blocksPerGridNew = (1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	const int shmem_size = sizeof(float)*THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}

	cudaSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float)));
    //timer.startEvent("PCGStep_Kernel1");
    PCGStep_Kernel1 << <blocksPerGrid, THREADS_PER_BLOCK, shmem_size >> >(input, state, parameters);
    //timer.endEvent();
	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif

	cudaSafeCall(cudaMemset(state.d_scanBeta, 0, sizeof(float)));
	//timer.startEvent("PCGStep_Kernel2");
	PCGStep_Kernel2 << <blocksPerGridNew, THREADS_PER_BLOCK, shmem_size >> >(input, state);
	//timer.endEvent();
	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif

	//timer.startEvent("PCGStep_Kernel3");
	PCGStep_Kernel3 << <blocksPerGridNew, THREADS_PER_BLOCK, shmem_size >> >(input, state);
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

	if (x < 1) {
		state.d_translation[0] = state.d_translation[0] + state.d_delta[0];
		state.d_angles[0] = state.d_angles[0] + state.d_deltaA[0];
	}
}

void ApplyLinearUpdate(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	const unsigned int N = 1; // This is different to all sparse solvers !!!
	//timer.startEvent("ApplyLinearUpdateDevice");
	ApplyLinearUpdateDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
    //timer.endEvent();
	cudaSafeCall(cudaDeviceSynchronize()); // Hm

	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif
}

/////////////////////////////////////////////////////////////////////////
// Transform Mesh
/////////////////////////////////////////////////////////////////////////

__global__ void ApplyTransformMeshDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.N; // Number of block variables
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N)
	{
		float3x3 R = evalRMat(state.d_angles[0]);
		state.d_mesh[x] = R*state.d_mesh[x] + state.d_translation[0];
	}
}

void TransformMesh(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer& timer)
{
	const unsigned int N = input.N; // This is different to all sparse solvers !!! number of vertices
	//timer.startEvent("Transform Mesh");
	ApplyTransformMeshDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);
	//timer.endEvent();
	cudaSafeCall(cudaDeviceSynchronize()); // Hm

	#ifdef _DEBUG
		cudaSafeCall(cudaDeviceSynchronize());
	#endif
}

////////////////////////////////////////////////////////////////////
// Main GN Solver Loop
////////////////////////////////////////////////////////////////////

extern "C" double ProcrustesSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverPerformanceSummary& stats)
{
    CUDATimer timer;
    timer.startEvent("Total");
	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
        timer.startEvent("Nonlinear Iteration");
        timer.startEvent("Nonlinear Setup");
		float residual = EvalResidual(input, state, parameters, timer);
		printf("%i: cost: %f\n", nIter, residual);

		Initialization(input, state, parameters, timer);
        timer.endEvent();

        timer.startEvent("Linear Solve");
		for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++) {
			PCGIteration(input, state, parameters, timer);
		}
        timer.endEvent();

        timer.startEvent("Nonlinear Finish");
		ApplyLinearUpdate(input, state, parameters, timer);
        timer.nextIteration();
        timer.endEvent();
        timer.endEvent();
	}

	float residual = EvalResidual(input, state, parameters, timer);
	printf("final cost: %f\n", residual);
    timer.endEvent();
    timer.evaluate(stats);
    return (double)residual;
}
