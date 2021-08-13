#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include "../../shared/cudaUtil.h"

#include "SFSSolverInput.h"
#include "../../shared/SolverIteration.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"



class CombinedSolver : public CombinedSolverBase {
private:
    std::shared_ptr<SimpleBuffer>   m_initialUnknown;
    std::shared_ptr<SimpleBuffer>   m_result;
    std::shared_ptr<SimpleBuffer>   m_initialLighting;
    std::shared_ptr<SimpleBuffer>   m_resultLighting;
    std::vector<unsigned int> m_dims;
public:
    CombinedSolver(const SFSSolverInput& inputGPU, CombinedSolverParameters params) : CombinedSolverBase("Shape And Shading")
	{
        m_combinedSolverParameters = params;
        m_initialUnknown = std::make_shared<SimpleBuffer>(*inputGPU.initialUnknown, true);
        m_result = std::make_shared<SimpleBuffer>(*inputGPU.initialUnknown, true);
        printf("reiniting lighting\n");
        m_initialLighting = std::make_shared<SimpleBuffer>(*inputGPU.lighting, true);
        m_resultLighting = std::make_shared<SimpleBuffer>(*inputGPU.lighting, true);
        printf("about to set params\n");
        inputGPU.setParameters(m_problemParams, m_resultLighting, m_result);
        printf("adding thallo solvers\n");
        m_dims = { (unsigned int)m_result->width(), (unsigned int)m_result->height(), 1 };

        addThalloSolvers(m_dims);
	}

    virtual void combinedSolveInit() override {
        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
    }

    virtual void preSingleSolve() override {
        resetGPUMemory();
    }
    virtual void postSingleSolve() override {}

    virtual void preNonlinearSolve(int) override {}
    virtual void postNonlinearSolve(int) override {}

    virtual void combinedSolveFinalize() override {
        ceresIterationComparison(m_name, m_combinedSolverParameters.thalloDoublePrecision);
    }

    std::shared_ptr<SimpleBuffer> result() {
        auto initBuffer = std::make_shared<SimpleBuffer>(*m_initialLighting, false);
        float* cpuInitLighting = (float*)initBuffer->data();
        auto resultBuffer = std::make_shared<SimpleBuffer>(*m_resultLighting, false);
        float* cpuFinalLighting = (float*)resultBuffer->data();
        printf("Lighting Result:\n");
        for (int i = 0; i < 9; ++i) {
            float init = cpuInitLighting[i];
            float result = cpuFinalLighting[i];
            printf("  %d: %g - %g = %g change\n", i, result, init, result - init);
        }
        printf(" Init: [ ");
        for (int i = 0; i < 9; ++i) {
            float init = cpuInitLighting[i];
            printf("%g", init);
            if (i < 8) {
                printf(", ");
            }
        }
        printf("]\n");
        printf(" Result: [ ");
        for (int i = 0; i < 9; ++i) {
            float result = cpuFinalLighting[i];
            printf("%g", result);
            if (i < 8) {
                printf(", ");
            }
        }
        printf("]\n");

        return m_result;
    }

    std::shared_ptr<SimpleBuffer> resultLighting() {
        return std::make_shared<SimpleBuffer>(*m_resultLighting, false);
    }
	void resetGPUMemory() {
        cudaSafeCall(cudaMemcpy(m_result->data(), m_initialUnknown->data(), m_dims[0]*m_dims[1]*sizeof(float), cudaMemcpyDeviceToDevice));
	}

};
