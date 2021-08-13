#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include "../../shared/cudaUtil.h"

#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"
#include "../../shared/SolverIteration.h"

#include "CUDAWarpingSolver.h"
#include "CUDAPatchSolverWarping.h"
#include "EigenSolverPoissonImageEditing.h"

class CombinedSolver : public CombinedSolverBase {
public:
    CombinedSolver(const ColorImageR32G32B32A32& image, const ColorImageR32G32B32A32& image1, const ColorImageR32& imageMask, CombinedSolverParameters params, bool useCUDAPatch, bool useEigen)
            : CombinedSolverBase("Poisson Image Editing") {
        m_combinedSolverParameters = params;
        m_useCUDAPatch = useCUDAPatch;
        m_useEigen = useEigen;
		m_image = image;
		m_image1 = image1;
		m_imageMask = imageMask;
        m_dims = { m_image.getWidth(), m_image.getHeight() };

        d_image     = createEmptyThalloImage(m_dims, ThalloImage::Type::FLOAT, 4, ThalloImage::Location::GPU, true);
        d_target    = createEmptyThalloImage(m_dims, ThalloImage::Type::FLOAT, 4, ThalloImage::Location::GPU, true);
        d_mask      = createEmptyThalloImage(m_dims, ThalloImage::Type::FLOAT, 1, ThalloImage::Location::GPU, true);

		resetGPUMemory();


        addSolver(std::make_shared<CUDAWarpingSolver>(m_dims), "Cuda", m_combinedSolverParameters.useCUDA);
        addSolver(std::make_shared<EigenSolverPoissonImageEditing>(m_dims), "Eigen", m_useEigen);
        addSolver(std::make_shared<CUDAPatchSolverWarping>(m_dims), "Cuda Patch", m_useCUDAPatch);
        addThalloSolvers(m_dims);
		
	}

    virtual void combinedSolveInit() override {
        m_problemParams.set("X", d_image);
        m_problemParams.set("T", d_target);
        m_problemParams.set("M", d_mask);

        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
    }
    virtual void preSingleSolve() override {
        resetGPUMemory();
    }
    virtual void postSingleSolve() override { 
        copyResultToCPU();
    }

    virtual void preNonlinearSolve(int) override {}
    virtual void postNonlinearSolve(int) override {}

    virtual void combinedSolveFinalize() override {
        ceresIterationComparison(m_name, m_combinedSolverParameters.thalloDoublePrecision);
    }

	void resetGPUMemory()
	{
        unsigned int N = m_image.getWidth()*m_image.getHeight();
        std::vector<float4> h_image(N);
        std::vector<float4> h_target(N);
        std::vector<float>  h_mask(N);

		for (unsigned int i = 0; i < m_image1.getHeight(); i++)
		{
			for (unsigned int j = 0; j < m_image1.getWidth(); j++)
			{
				ml::vec4f v = m_image(j, i);
				h_image[i*m_image.getWidth() + j] = make_float4(v.x, v.y, v.z, 255);

				ml::vec4f t = m_image1(j, i);
				h_target[i*m_image.getWidth() + j] = make_float4(t.x, t.y, t.z, 255);

				if (m_imageMask(j, i) == 255) h_mask[i*m_image.getWidth() + j] = 0;
				else						  h_mask[i*m_image.getWidth() + j] = 255;
			}
		}
        d_image->update(h_image);
        d_target->update(h_target);
        d_mask->update(h_mask);
	}
    ColorImageR32G32B32A32* result() {
        return &m_result;
    }

	void copyResultToCPU() {
		m_result = ColorImageR32G32B32A32(m_image.getWidth(), m_image.getHeight());
        cudaSafeCall(cudaMemcpy(m_result.getData(), d_image->data(), sizeof(float4)*m_image.getWidth()*m_image.getHeight(), cudaMemcpyDeviceToHost));
	}

private:

    std::vector<unsigned int> m_dims;

	ColorImageR32G32B32A32 m_image;
	ColorImageR32G32B32A32 m_image1;
	ColorImageR32		   m_imageMask;

	ColorImageR32G32B32A32 m_result;

	std::shared_ptr<ThalloImage> d_image;
    std::shared_ptr<ThalloImage> d_target;
    std::shared_ptr<ThalloImage> d_mask;
	
    bool m_useCUDAPatch;
    bool m_useEigen;
};
