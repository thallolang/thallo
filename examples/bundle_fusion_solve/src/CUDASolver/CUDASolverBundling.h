#pragma once

#include "SolverBundlingState.h"
#include "SolverBundlingParameters.h"
#define HAS_CUTIL
#include "SolverBase.h"

class CUDASolverBundling : public SolverBase
{
public:
    CUDASolverBundling(const SolverInput& input, const SolverState& state, const SolverParameters& parameters);
    ~CUDASolverBundling();
    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, SolverPerformanceSummary& stats, bool profileSolve, std::vector<SolverIteration>& iter) override;
protected:
    SolverInput m_input;
    SolverState m_state;
    SolverParameters m_parameters;
    SolverStateAnalysis m_solverExtra;
    unsigned int THREADS_PER_BLOCK = 512;
};
