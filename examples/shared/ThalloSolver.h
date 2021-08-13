#pragma once

#include <cassert>

extern "C" {
#include "Thallo.h"
}
#include "ThalloUtils.h"
#include "CudaArray.h"
#include "SolverIteration.h"
#include "cudaUtil.h"
#include "NamedParameters.h"
#include "SolverBase.h"
#include <cstdio>
#include <cstring>

static NamedParameters copyParametersAndConvertUnknownsToDouble(const NamedParameters& original) {
    NamedParameters newParams(original);
    std::vector<NamedParameters::Parameter> unknownParameters = original.unknownParameters();
    for (auto p : unknownParameters) {
        auto gpuDoubleImage = copyImageTo(getDoubleImageFromFloatImage(copyImageTo(p.im, ThalloImage::Location::CPU)), ThalloImage::Location::GPU);
        newParams.set(p.name, gpuDoubleImage);
    }
    return newParams;
}

static void copyUnknownsFromDoubleToFloat(const NamedParameters& floatParams, const NamedParameters& doubleParams) {
    std::vector<NamedParameters::Parameter> unknownParameters = doubleParams.unknownParameters();
    for (auto p : unknownParameters) {
        auto cpuDoubleImage = copyImageTo(p.im, ThalloImage::Location::CPU);
        auto cpuFloatImage = getFloatImageFromDoubleImage(cpuDoubleImage);
        NamedParameters::Parameter param;
        floatParams.get(p.name, param);
        copyImage(param.im, cpuFloatImage);
    }
}



class ThalloSolver : public SolverBase {

public:
    ThalloSolver(const std::vector<unsigned int>& dimensions, const std::string& terraFile, const std::string& thalloName, bool doublePrecision = false, bool invasiveTiming = false, int autoscheduled = 0, bool thalloCPU = false) 
        : m_optimizerState(nullptr), m_problem(nullptr), m_plan(nullptr), m_doublePrecision(doublePrecision)
	{
        Thallo_InitializationParameters initParams;
        memset(&initParams, 0, sizeof(Thallo_InitializationParameters));
        initParams.verbosityLevel = 1;
        initParams.timingLevel = invasiveTiming ? 2 : 1;
        initParams.doublePrecision = (int)doublePrecision;
        initParams.useAutoscheduler = autoscheduled;

#ifdef THALLO_CPU // Force CPU Mode
        thalloCPU = true;
#endif
        initParams.cpuOnly = (int)thalloCPU;

        printf("Thallo Solver Init\n");
        m_optimizerState = Thallo_NewState(initParams);
        printf("Thallo Solver Define\n");
		m_problem = Thallo_ProblemDefine(m_optimizerState, terraFile.c_str(), thalloName.c_str());
        printf("Thallo Solver Plan\n");

	// TODO: track down why this was inserted and fix it...
	static unsigned int static_dims[10] = {};
	for (int i = 0; i < dimensions.size(); ++i) {
	  static_dims[i] = dimensions[i];
	}
	
        m_plan = Thallo_ProblemPlan(m_optimizerState, m_problem, static_dims);
		assert(m_optimizerState);
		assert(m_problem);
		assert(m_plan);
	}

    ~ThalloSolver()
	{
		if (m_plan) {
			Thallo_PlanFree(m_optimizerState, m_plan);
		}

		if (m_problem) {
			Thallo_ProblemDelete(m_optimizerState, m_problem);
		}
	}

    virtual double solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, SolverPerformanceSummary& perfStats, bool profiledSolve, std::vector<SolverIteration>& iters) override {
        NamedParameters finalProblemParameters = problemParameters;
        if (m_doublePrecision) {
            finalProblemParameters = copyParametersAndConvertUnknownsToDouble(problemParameters);
        }
        setAllSolverParameters(m_optimizerState, m_plan, solverParameters);
        if (profiledSolve) {
            launchProfiledSolve(m_optimizerState, m_plan, finalProblemParameters.data().data(), iters);
        } else {
            Thallo_ProblemSolve(m_optimizerState, m_plan, finalProblemParameters.data().data());
        }
        m_finalCost = Thallo_ProblemCurrentCost(m_optimizerState, m_plan);
        // TODO: Accumulate statistics over multiple solves
        Thallo_GetPerformanceSummary(m_optimizerState, m_plan, (Thallo_PerformanceSummary*)&m_summaryStats);
        if (m_doublePrecision) {
            copyUnknownsFromDoubleToFloat(problemParameters, finalProblemParameters);
        }

        return m_finalCost;
	}

	Thallo_State*		m_optimizerState;
	Thallo_Problem*	m_problem;
	Thallo_Plan*		m_plan;
    bool m_doublePrecision = false;
};
