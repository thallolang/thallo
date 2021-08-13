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
    CombinedSolver(std::unordered_map<std::string, DepthImage32>& inputs, const CombinedSolverParameters& params) : CombinedSolverBase("Spatially Varying Deconvolution") {
        m_combinedSolverParameters = params;

        uint W = inputs["M"].getDimX();
        uint H = inputs["M"].getDimY();
        auto K = inputs["K"];
        const unsigned int kernelWidth = 17;
        std::vector<unsigned int> unwrappedKernelDims = { K.getWidth(), K.getHeight() };
        alwaysAssertM(K.getWidth() % kernelWidth == 0 && K.getHeight() % kernelWidth == 0, "Kernel must be size 17");
        std::vector<unsigned int> kernelCounts = { K.getWidth() / kernelWidth, K.getHeight() / kernelWidth };
        std::vector<unsigned int> kernelDims{ kernelWidth, kernelWidth, kernelCounts[0] * kernelCounts[1] };
        unsigned int kSize = kernelDims[0] * kernelDims[1];

        { // Stack kernels, add sparse connections from image to layer of graph
            auto im = inputs["K"];
            std::vector<unsigned int> unwrappedKernelDims = { im.getWidth(), im.getHeight() };
            auto thalloIm = createEmptyThalloImage(unwrappedKernelDims, ThalloImage::Type::FLOAT, 1, ThalloImage::CPU, false);
            updateThalloImage(thalloIm, im);
            float* kData = (float*)thalloIm->data();
            std::vector<float> stackedData(kernelDims[0] * kernelDims[1] * kernelDims[2]);
            
            for (unsigned int ky = 0; ky < unwrappedKernelDims[1]; ++ky) {
                int y = ky % kernelWidth;
                int kIndexY = ky / kernelWidth;
                for (unsigned int kx = 0; kx < unwrappedKernelDims[0]; ++kx) {
                    int x = kx % kernelWidth;
                    int kIndexX = kx / kernelWidth;
                    int kernelIndex = kIndexY*kernelCounts[0] + kIndexX;
                    stackedData[kernelIndex*kSize + kernelWidth*y + x] = kData[ky*unwrappedKernelDims[0] + kx];
                }
            }
            ColorImageR32 kernel_out(kernelDims[0], kernelDims[1]);
            for (int kIndexY = 0; kIndexY < kernelCounts[1]; ++kIndexY) {
                for (int kIndexX = 0; kIndexX < kernelCounts[0]; ++kIndexX) {
                    int kernelIndex = kIndexY*kernelCounts[0] + kIndexX;
                    int off = kernelIndex*kSize;
                    float kSum = 0.0;
                    for (int y = 0; y < kernelDims[1]; ++y) {
                        for (int x = 0; x < kernelDims[0]; ++x) {
                            float p = stackedData[off + kernelWidth*y + x];
                            kSum += p;
                            kernel_out.setPixel(x,y,p);
                        }
                    }
                    float newKSum = 0.0;
                    for (int y = 0; y < kernelDims[1]; ++y) {
                        for (int x = 0; x < kernelDims[0]; ++x) {
                            float p = stackedData[off + kernelWidth*y + x] / kSum;
                            kernel_out.setPixel(x, y, p);
                            newKSum += p;
                            stackedData[off + kernelWidth*y + x] = p;
                        }
                    }
                    

                    printf("Kernel total was: %f, now: %f\n", kSum, newKSum);
                    char buffer[50];
                    sprintf(buffer, "k_%d_%d.tif", kIndexX, kIndexY);
                    FreeImageWrapper::saveImage(buffer, kernel_out);
                }
            }


            m_stackedKernels = createEmptyThalloImage(kernelDims, ThalloImage::Type::FLOAT, 1, ThalloImage::GPU, false);
            m_stackedKernels->update(stackedData);
        }

        {
            std::vector<int> kIndices(W*H);
            float kSectorWidth = W / (float)kernelCounts[0];
            float kSectorHeight = H / (float)kernelCounts[1];
            for (int y = 0; y < (int)H; ++y) {
                int kIndexY = (int)(y / kSectorHeight);
                alwaysAssertM(kIndexY < (int)kernelCounts[1], "kernel sectors too short");
                for (int x = 0; x < (int)W; ++x) {
                    int kIndexX = (int)(x / kSectorWidth);
                    alwaysAssertM(kIndexX < (int)kernelCounts[0], "kernel sectors too short");
                    kIndices[y*W + x] = kIndexY*kernelCounts[0] + kIndexX;
                }
            }
            m_imageToKernelIndex = std::make_shared<ThalloGraph>(std::vector<std::vector<int>>({ kIndices }));
        }


        m_dims = { W, H, kernelWidth, kernelDims[2] };
        m_initial = inputs["x0"];

        m_result = ColorImageR32(m_initial);
        m_unknown   = createEmptyThalloImage({ W, H }, ThalloImage::Type::FLOAT, 1, ThalloImage::GPU, true);

        auto depthToThalloIm = [](DepthImage32& im) {
            auto thalloIm = createEmptyThalloImage({ im.getWidth(), im.getHeight() }, ThalloImage::Type::FLOAT, 1, ThalloImage::GPU, false);
            updateThalloImage(thalloIm, im);
            return thalloIm;
        };
        m_imageParamNames = { "M", "b_1", "b_2", "b_3" }; 
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
        m_problemParams.set("K", m_stackedKernels);
        m_problemParams.set("G", m_imageToKernelIndex);
        
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
    std::shared_ptr<ThalloImage> m_stackedKernels;
    std::shared_ptr<ThalloGraph> m_imageToKernelIndex;
    std::vector<std::string> m_imageParamNames;
    ColorImageR32 m_result;
	

};
