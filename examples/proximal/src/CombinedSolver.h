#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>

#include "../../shared/CombinedSolverBase.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/SolverIteration.h"


template <typename T>
static void updateThalloImage(std::shared_ptr<ThalloImage> dst, BaseImage<T> src) {
    dst->update((void*)src.getData(), sizeof(T)*src.getWidth()*src.getHeight(), ThalloImage::Location::CPU);
}

class CombinedSolver : public CombinedSolverBase {
public:
    CombinedSolver(std::unordered_map<std::string, DepthImage32>& inputs, const CombinedSolverParameters& params) : CombinedSolverBase("Deconvolution") {
        m_combinedSolverParameters = params;

        uint W = inputs["M"].getDimX();
        uint H = inputs["M"].getDimY();
        uint Kd = inputs["K"].getDimY();
        alwaysAssertM(Kd == 15 && Kd == inputs["K"].getDimX() && Kd == inputs["K"].getDimY(), "Kernel must be size 15");

	m_dims = { W, H, Kd };
        m_initial = inputs["x0"];

        m_result = ColorImageR32(m_initial);
        m_unknown   = createEmptyThalloImage({ W, H }, ThalloImage::Type::FLOAT, 1, ThalloImage::GPU, true);

        auto depthToThalloIm = [](DepthImage32& im) {
            auto thalloIm = createEmptyThalloImage({ im.getWidth(), im.getHeight() }, ThalloImage::Type::FLOAT, 1, ThalloImage::GPU, false);
            updateThalloImage(thalloIm, im);
            return thalloIm;
        };
        m_imageParamNames = { "M", "b_1", "b_2", "b_3", "K" }; 
        for (auto s : m_imageParamNames) {
            m_inputImages[s] = depthToThalloIm(inputs[s]);
        }
        auto lambda = inputs["lambda"].getData();
        printf("Lambda 1: %g\n", lambda[0]);
        printf("Lambda 2: %g\n", lambda[1]);
        m_lambda1Sqrt = sqrtf(lambda[0]);
        m_lambda2Sqrt = sqrtf(lambda[1]);

	resetGPU();
        // Adds Thallo solvers according to settings in m_combinedSolverParameters
        addThalloSolvers(m_dims);
	}


    virtual void combinedSolveInit() override {
        // Set in the same order as indices in param declaration
        m_problemParams.set("sqrt_l1", &m_lambda1Sqrt); 
        m_problemParams.set("sqrt_l2", &m_lambda2Sqrt);
        m_problemParams.set("X", m_unknown);
        for (auto s : m_imageParamNames) {
            m_problemParams.set(s, m_inputImages[s]);
        }
        
        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
    }

    virtual void preSingleSolve() override {
        
    }
    virtual void postSingleSolve() override {}
    virtual void preNonlinearSolve(int) override {
      resetGPU();
    }
    virtual void postNonlinearSolve(int) override {}

    virtual void combinedSolveFinalize() override {
        cudaSafeCall(cudaMemcpy(m_result.getData(), m_unknown->data(), sizeof(float)*m_dims[0] * m_dims[1], cudaMemcpyDeviceToHost));
    }

    void resetGPU() {
        updateThalloImage(m_unknown, m_initial);
    }

    BaseImage<float> result() {
        return m_result;
    }

private:

    float m_lambda1Sqrt, m_lambda2Sqrt;

    std::vector<uint> m_dims;
    DepthImage32 m_initial;
    std::unordered_map<std::string, std::shared_ptr<ThalloImage>> m_inputImages;
    std::shared_ptr<ThalloImage> m_unknown;
    std::vector<std::string> m_imageParamNames;
    ColorImageR32 m_result;
	

};
