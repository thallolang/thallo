#pragma once

#include <cuda_runtime.h>

#include "../../shared/cudaUtil.h"
#include "WarpingSolverParameters.h"
#include "WarpingSolverState.h"
#include "../../shared/SolverBase.h"

class CUDAWarpingSolver : public SolverBase
{
	public:
        CUDAWarpingSolver(unsigned int N);
		~CUDAWarpingSolver();

        virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, SolverPerformanceSummary& stats, bool profileSolve, std::vector<SolverIteration>& iter) override;
		
	private:
        SolverInput m_solverInput;
		SolverState	m_solverState;
        unsigned int m_N;
};
