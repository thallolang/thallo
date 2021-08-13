#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>
#include "../../shared/cudaUtil.h"
#include "CUDAWarpingSolver.h"
#include "../../shared/OpenMesh.h"
#include "../../shared/SolverIteration.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"
#include "../../shared/ThalloGraph.h"
#include <cuda_profiler_api.h>
#include "RotationHelper.h"

class CombinedSolver : public CombinedSolverBase
{
	public:
		CombinedSolver(const SimpleMesh* mesh, const SimpleMesh* target, CombinedSolverParameters params) : CombinedSolverBase("Procrustes Alignment")
		{
			m_result = *mesh;
			m_target = *target;
			m_initial = m_result;
            m_combinedSolverParameters = params;

			unsigned int N = (unsigned int)mesh->n_vertices();

            m_dims = { N, 1 };
            m_vertexPosFloat3 = createEmptyThalloImage({ N }, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, false);
            m_targetPosFloat3 = createEmptyThalloImage({ N }, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, false);

			m_angleFloat3				= createEmptyThalloImage({ 1 }, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, true);
			m_translationFloat3			= createEmptyThalloImage({ 1 }, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, true);

			resetGPUMemory();
            
            addSolver(std::make_shared<CUDAWarpingSolver>(N), "Cuda", m_combinedSolverParameters.useCUDA);
            addThalloSolvers(m_dims);
		} 

        virtual void combinedSolveInit() override {
			m_problemParams.set("Translation", m_translationFloat3);
            m_problemParams.set("Angle", m_angleFloat3);

            m_problemParams.set("Mesh", m_vertexPosFloat3);
            m_problemParams.set("Target", m_targetPosFloat3);

            m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
            m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
        }

        virtual void preNonlinearSolve(int i) override {}

        virtual void postNonlinearSolve(int) override {}

        virtual void preSingleSolve() override {
            m_result = m_initial;
            resetGPUMemory();
        }
        virtual void postSingleSolve() override {
            copyResultToCPUFromFloat3();
        }
        virtual void combinedSolveFinalize() override {}

		void resetGPUMemory()
		{
			unsigned int N = (unsigned int)m_initial.n_vertices();

			std::vector<float3> h_vertexPosFloat3(N);
			std::vector<float3> h_targetPosFloat3(N);
			
			for (unsigned int i = 0; i < N; i++)
			{
				const Vec3f& pt = m_initial.point(VertexHandle(i));
				const Vec3f& pTarget = m_target.point(VertexHandle(i));

				h_vertexPosFloat3[i] = make_float3(pt[0], pt[1], pt[2]);
				h_targetPosFloat3[i] = make_float3(pTarget[0], pTarget[1], pTarget[2]);
			}
			

			// Angles
			std::vector<float3> h_angles(1);
			h_angles[0] = make_float3(0.0f, 0.0f, 0.0f);
            m_angleFloat3->update(h_angles);
            
			// Translation
			std::vector<float3> h_translations(1);
			h_translations[0] = make_float3(0.0f, 0.0f, 0.0f);
			m_translationFloat3->update(h_translations);
			
			
			m_vertexPosFloat3->update(h_vertexPosFloat3);
            m_targetPosFloat3->update(h_targetPosFloat3);
		}

        ~CombinedSolver()
		{
		}

        SimpleMesh* result() {
            return &m_result;
        }

		void copyResultToCPUFromFloat3()
		{
			unsigned int N = (unsigned int)m_result.n_vertices();
			std::vector<float3> h_vertexPosFloat3(N), h_translation(1), h_rotation(1);
            m_vertexPosFloat3->copyTo(h_vertexPosFloat3);
            m_translationFloat3->copyTo(h_translation);
            m_angleFloat3->copyTo(h_rotation);
            float3x3 R = evalRMat(h_rotation[0]);
			for (unsigned int i = 0; i < N; i++)
			{
                float3 v = R*h_vertexPosFloat3[i] + h_translation[0];
                m_result.set_point(VertexHandle(i), Vec3f(v.x, v.y, v.z));
			}
		}

	private:

		SimpleMesh m_result;
		SimpleMesh m_target;
		SimpleMesh m_initial;

        std::vector<unsigned int> m_dims;
	
        std::shared_ptr<ThalloImage> m_vertexPosFloat3;
		std::shared_ptr<ThalloImage> m_targetPosFloat3;

        std::shared_ptr<ThalloImage> m_angleFloat3;
		std::shared_ptr<ThalloImage> m_translationFloat3;
};
