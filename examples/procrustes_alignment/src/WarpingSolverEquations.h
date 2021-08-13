#pragma once

#include "../../shared/cudaUtil.h"

#include "WarpingSolverUtil.h"
#include "WarpingSolverState.h"
#include "WarpingSolverParameters.h"
#include "RotationHelper.h"

////////////////////////////////////////
// evalF
////////////////////////////////////////

__inline__ __device__ float evalFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters) {
	float3 e = make_float3(0.0f, 0.0f, 0.0f);
	if (state.d_target[variableIdx].x != MINF){
		float3x3 R = evalRMat(state.d_angles[0]);
		float3 vTrans = R*state.d_mesh[variableIdx] + state.d_translation[0];
		float3 e_fit = vTrans - state.d_target[variableIdx];
		e += e_fit*e_fit;
	}
	float res = e.x + e.y + e.z;
	return res;
}

////////////////////////////////////////
// applyJT : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float3 evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float3& outAngle)
{
	mat3x1 b;  b.setZero();
	mat3x1 bA; bA.setZero();
	mat3x1 v = mat3x1(state.d_mesh[variableIdx]);
	mat3x1 target = mat3x1(state.d_target[variableIdx]);
	mat3x3 R = evalRMat(state.d_angles[0]);
	mat3x1 vTrans = R*v + mat3x1(state.d_translation[0]);

	mat3x3 dRAlpha, dRBeta, dRGamma;
	evalDerivativeRotationMatrix(state.d_angles[0], dRAlpha, dRBeta, dRGamma);
	mat3x3 D = evalDerivativeRotationTimesVector(dRAlpha, dRBeta, dRGamma, v);
	
	// fit
	if (target(0) != MINF) {
		b	-= 2.0f * (vTrans - target);
		bA	-= 2.0f * D.getTranspose()*(vTrans - target);
	}
	outAngle = bA;
	return b;
}

////////////////////////////////////////
// applyJTJ : this function is called per variable and evaluates each residual influencing that variable (i.e., each energy term per variable)
////////////////////////////////////////

__inline__ __device__ float3 applyJTJDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float3& outAngle) {
	mat3x1 b; b.setZero();
	mat3x1 bA; bA.setZero();
	mat3x1 p = mat3x1(state.d_p[0]);
	mat3x1 pA = mat3x1(state.d_pA[0]);

	mat3x1 v = mat3x1(state.d_mesh[variableIdx]);
	mat3x1 target = mat3x1(state.d_target[variableIdx]);
	mat3x3 R = evalRMat(state.d_angles[0]);

	mat3x3 dRAlpha, dRBeta, dRGamma;
	evalDerivativeRotationMatrix(state.d_angles[0], dRAlpha, dRBeta, dRGamma);
	mat3x3 D = evalDerivativeRotationTimesVector(dRAlpha, dRBeta, dRGamma, v);
	// fit/pos
	if (state.d_target[variableIdx].x != MINF) {
		mat3x1 tmpJ		  = D*pA + p;
		b  += 2.0f*tmpJ;
		bA += 2.0f*D.getTranspose()*tmpJ;
	}
	
	outAngle = bA;
	return b;
}
