#pragma once
#include <nanoflann/include/nanoflann.hpp>
#include "mLibInclude.h"
#include <stdio.h>
#include "../../shared/cudaUtil.h"
#include "CUDAWarpingSolver.h"
#define GLOG_NO_ABBREVIATED_SEVERITIES
#include "../../shared/ThalloUtils.h"

#include "ceres/ceres.h"
#include "glog/logging.h"
#include "ceres/rotation.h"
#include "../../shared/CeresHelpers.h"
#include "../../shared/OpenMesh.h"
#include "../../shared/SolverIteration.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"
#include "../../shared/ThalloGraph.h"
#include <cuda_profiler_api.h>
#include <Eigen/Core>
#include <Eigen/Geometry> 
#include <Eigen/StdVector>

using namespace nanoflann;
// For nanoflann computation
struct PointCloud_nanoflann
{
	std::vector<float3>  pts;

	// Must return the number of data points
	inline size_t kdtree_get_point_count() const { return pts.size(); }

	// Returns the distance between the vector "p1[0:size-1]" and the data point with index "idx_p2" stored in the class:
	inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t /*size*/) const
	{
		const float d0 = p1[0] - pts[idx_p2].x;
		const float d1 = p1[1] - pts[idx_p2].y;
		const float d2 = p1[2] - pts[idx_p2].z;
		return d0*d0 + d1*d1 + d2*d2;
	}

	// Returns the dim'th component of the idx'th point in the class:
	// Since this is inlined and the "dim" argument is typically an immediate value, the
	//  "if/else's" are actually solved at compile time.
	inline float kdtree_get_pt(const size_t idx, int dim) const
	{
		if (dim == 0) return pts[idx].x;
		else if (dim == 1) return pts[idx].y;
		else return pts[idx].z;
	}

	// Optional bounding-box computation: return false to default to a standard bbox computation loop.
	//   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
	//   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

};

typedef KDTreeSingleIndexAdaptor<
	L2_Simple_Adaptor<float, PointCloud_nanoflann>,
	PointCloud_nanoflann,
	3 /* dim */
> NanoKDTree;


static bool operator==(const float3& v0, const float3& v1) {
	return (v0.x == v1.x) && (v0.y == v1.y) && (v0.z == v1.z);
}
static bool operator!=(const float3& v0, const float3& v1) {
	return !(v0 == v1);
}
#define MAX_K 20

static float clamp(float v, float mn, float mx) {
	return std::max(mn, std::min(v, mx));
}

static float dot(vec2T<float> v0, vec2T<float> v1) {
    return v0.x*v0.x + v0.y*v0.y;
}

// The principal point is not modeled
static vec2T<float> snavely_projection(vec3T<float> point, vec3T<float> translation, vec3T<float> rodriquesVector, float focalLength, float l_1, float l_2) {
    vec3T<float> p;
    ceres::AngleAxisRotatePoint((float*)&rodriquesVector, (float*)&point, (float*)&p);
    p += translation;
    // the camera coordinate system has a negative z axis.
    vec2T<float> center_of_distortion = vec2T<float>(-p.x / p.z, -p.y / p.z);
    //printf("p.z = %g\n", p.z);

    //Apply second and fourth order radial distortion.
    float r2 = dot(center_of_distortion, center_of_distortion);
    float distortion = 1.0f + r2  * (l_1 + l_2  * r2);
    // Compute final projected point position.
    return center_of_distortion * focalLength * distortion;
}

struct CameraParameters {
    vec3T<float> rot = { 0.1, 0, 0 };
    vec3T<float> pos = {0,0,0};
    float focalLength = 5903.7;
    float l_1 = 0.109306;
    float l_2 = -1.43375;
};


class CombinedSolver : public CombinedSolverBase
{
	public:
        CombinedSolver(const SimpleMesh* averageMesh, const SimpleMesh* target, Eigen::MatrixXf* blendshapes, CombinedSolverParameters params) : CombinedSolverBase("Face Fitting")
		{
			m_result = *averageMesh;
			m_target = *target;
			m_blendshapes = *blendshapes;
			m_averageMesh = *averageMesh;
            m_combinedSolverParameters = params;

			unsigned int N = (unsigned int)m_averageMesh.n_vertices();
            unsigned int M = (unsigned int)m_blendshapes.cols();

			m_vertexPos3	  = createEmptyThalloImage({ N }, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, false);
			m_targetPos3	  = createEmptyThalloImage({ N }, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, false);
			m_averageMeshPos3 = createEmptyThalloImage({ N }, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, false);

			m_dimsBasis = { N, M, 1 };
			m_blendshapeBasis = createEmptyThalloImage({ N, M }, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, false);
			m_blendshapeWeights = createEmptyThalloImage({ M }, ThalloImage::Type::FLOAT, 1, ThalloImage::GPU, true);
            m_regSqrt = sqrtf(0.001f);

            m_camParamsHost.pos = { 0, 0, 5 };
            m_camParams = createEmptyThalloImage({ N }, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, false);

            m_targetPos2 = createEmptyThalloImage({ N }, ThalloImage::Type::FLOAT, 2, ThalloImage::GPU, false);
   
			resetGPUMemory();

			m_target.request_face_normals();
			m_target.request_vertex_normals();
			m_target.update_normals();

            addSolver(std::make_shared<CUDAWarpingSolver>(N, M), "Cuda", m_combinedSolverParameters.useCUDA);
            addThalloSolvers(m_dimsBasis);
		} 

        virtual void combinedSolveInit() override {
			m_problemParams.set("BlendshapeWeights", m_blendshapeWeights);
			m_problemParams.set("AverageMesh",		 m_averageMeshPos3);
			m_problemParams.set("BlendshapeBasis",	 m_blendshapeBasis);
            m_problemParams.set("Mesh",			     m_vertexPos3);
            m_problemParams.set("Target",			 m_targetPos2);
            m_problemParams.set("w_regSqrt", &m_regSqrt);
            m_problemParams.set("CamParams", m_camParams);

            m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
            m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
        }

        virtual void preNonlinearSolve(int i) override {
			//setConstraints(0.01f, 0.70f);
		}

        virtual void postNonlinearSolve(int i) override { 
            computeResultOnCPUFromUnknowns();
		}

        virtual void preSingleSolve() override {
            m_result = m_averageMesh;
			m_result.request_face_normals();
			m_result.request_vertex_normals();
			m_result.update_normals();

			m_targetAccelerationStructure = generateAccelerationStructure(m_target);

			resetGPUMemory();
        }
        virtual void postSingleSolve() override {
            computeResultOnCPUFromUnknowns();
        }

        virtual void combinedSolveFinalize() override {
            if (m_combinedSolverParameters.profileSolve) {
                ceresIterationComparison(m_name, m_combinedSolverParameters.thalloDoublePrecision);
            }
        }

		void setConstraints(float positionThreshold = std::numeric_limits<float>::infinity(), float normalThreshold = 0.0f)
		{
			m_result.update_normals();

			unsigned int N = (unsigned int)m_result.n_vertices();
			std::vector<float3> h_vertexPosTargetFloat3(N);
			float3 invalidPt = make_float3(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());

			unsigned int validCorres = 0;
			for (int i = 0; i < (int)N; i++)
			{
				std::vector<size_t> neighbors(1);
				std::vector<float>  out_dist_sqr(1);
				auto currentPt = m_result.point(VertexHandle(i));
				m_targetAccelerationStructure->knnSearch(currentPt.data(), 1, &neighbors[0], &out_dist_sqr[0]);
				const Vec3f target = m_target.point(VertexHandle((int)neighbors[0]));
				bool isBoundary = m_target.is_boundary(VertexHandle((int)neighbors[0]));
				float dist = (target - currentPt).length();
				const Vec3f meshNormal = m_result.normal(VertexHandle(i));
				const Vec3f targetNormal = m_target.normal(VertexHandle((int)neighbors[0]));

				if (dist <= positionThreshold && !isBoundary && OpenMesh::dot(meshNormal, targetNormal) > normalThreshold)
				{
					h_vertexPosTargetFloat3[i] = make_float3(target[0], target[1], target[2]);
					validCorres++;
				}
				else
				{
					h_vertexPosTargetFloat3[i] = invalidPt;
				}
			}

			m_targetPos3->update(h_vertexPosTargetFloat3);

            std::vector<float2> h_vertexPosTargetFloat2(N);
            float2 invalidPt2 = make_float2(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
            auto C = m_camParamsHost;
            for (int i = 0; i < (int)N; ++i) {
                auto p = h_vertexPosTargetFloat3[i];
                auto newP = snavely_projection({ p.x, p.y, p.z }, C.pos, C.rot, C.focalLength, C.l_1, C.l_2);
                float2 newP2 = { newP.x, newP.y };
                h_vertexPosTargetFloat2[i] = (p.x > -9999999.9f) ? newP2 : invalidPt2;
            }
            m_targetPos2->update(h_vertexPosTargetFloat2);

			std::cout << "Number of Correspondences: " << validCorres << " out of " << N << std::endl;
		}

		void resetGPUMemory()
		{
            unsigned int N = m_dimsBasis[0];
            unsigned int M = m_dimsBasis[1];

			std::vector<float3> h_vertexPosFloat3(N);
			std::vector<float3> h_targetPosFloat3(N);
			std::vector<float3> h_averageMeshPosFloat3(N);
			std::vector<float3> h_blendshapeBasis(N*M);
			std::vector<float>  h_blendshapeWeights(M);
			
			for (unsigned int i = 0; i < N; i++)
			{
				const Vec3f& pt = m_averageMesh.point(VertexHandle(i));
				const Vec3f& pTarget = m_target.point(VertexHandle(i));

				h_vertexPosFloat3[i] = make_float3(pt[0], pt[1], pt[2]);
				h_averageMeshPosFloat3[i] = make_float3(pt[0], pt[1], pt[2]);
				h_targetPosFloat3[i] = make_float3(pTarget[0], pTarget[1], pTarget[2]);

				for (unsigned int j = 0; j < M; j++)
				{
					h_blendshapeBasis[j*N + i] = make_float3(m_blendshapes(3 * i + 0, j), m_blendshapes(3 * i + 1, j), m_blendshapes(3 * i + 2, j));
				}
			}

			for (unsigned int j = 0; j < M; j++)
			{
				h_blendshapeWeights[j] = 0.0f;
			}

            //h_blendshapeWeights[0] = 0.1f;
            //h_blendshapeWeights[70] = 2.0f;


            std::vector<vec2T<float>> h_target(N);
            auto C = m_camParamsHost;
            for (unsigned int i = 0; i < N; i++) {
                h_target[i] = snavely_projection(h_targetPosFloat3[i], C.pos, C.rot, C.focalLength, C.l_1, C.l_2);
            }

            {
                std::ofstream outTar("target2D.csv");
                outTar << std::scientific;
                outTar << std::setprecision(20);
                outTar << "x,y" << std::endl;
                for (int i = 0; i < N; ++i) {
                    outTar << h_target[i].x << ", " << h_target[i].y << std::endl;
                }
            }

            float minX = std::numeric_limits<float>::infinity();
            float minY = std::numeric_limits<float>::infinity();
            float maxX = -std::numeric_limits<float>::infinity();
            float maxY = -std::numeric_limits<float>::infinity();
            for (auto p : h_target) {
                minX = std::min(p.x, minX);
                maxX = std::max(p.x, maxX);
                minY = std::min(p.y, minY);
                maxY = std::max(p.y, maxY);
            }
            printf("min point: (%g, %g)\nmax point: (%g, %g)\n", minX, minY, maxX, maxY);
			
			m_blendshapeBasis->update(h_blendshapeBasis);
			m_blendshapeWeights->update(h_blendshapeWeights);
			m_vertexPos3->update(h_vertexPosFloat3);
			m_averageMeshPos3->update(h_vertexPosFloat3);
            m_targetPos3->update(h_targetPosFloat3);
            m_targetPos2->update(h_target);


            float* camParams = (float*)&m_camParamsHost;
            for (int i = 0; i < 9; ++i) {
                printf("%i: %g\n", i, camParams[i]);
            }
            m_camParams->update(&m_camParamsHost, sizeof(CameraParameters), ThalloImage::CPU);
            computeResultOnCPUFromUnknowns();
		}

        ~CombinedSolver() {
		}

        SimpleMesh* result() {
            return &m_result;
        }


        void computeResultOnCPUFromUnknowns()
        {
            unsigned int N = m_dimsBasis[0];
            unsigned int M = m_dimsBasis[1];
            std::vector<float3> h_vertexPosFloat3(N);

            std::vector<float3> h_averageMeshPos3(N);
            std::vector<float3> h_blendshapeBasis(N*M);
            std::vector<float> h_blendshapeWeights(M);
            m_averageMeshPos3->copyTo(h_averageMeshPos3);
            m_blendshapeBasis->copyTo(h_blendshapeBasis);
            m_blendshapeWeights->copyTo(h_blendshapeWeights);
            for (unsigned int i = 0; i < N; i++)
            {
                float3 result = h_averageMeshPos3[i];
                for (unsigned m = 0; m < M; ++m) {
                    result += h_blendshapeBasis[m*N + i] * h_blendshapeWeights[m];
                }
                m_result.set_point(VertexHandle(i), Vec3f(result.x, result.y, result.z));
            }
        }

	private:

		std::unique_ptr<NanoKDTree> generateAccelerationStructure(const SimpleMesh& mesh) {
			unsigned int N = (unsigned int)mesh.n_vertices();

			m_pointCloud.pts.resize(N);
			for (unsigned int i = 0; i < N; i++)
			{
				auto p = m_target.point(VertexHandle(i));
				m_pointCloud.pts[i] = { p[0], p[1], p[2] };
			}
			std::unique_ptr<NanoKDTree> tree = std::unique_ptr<NanoKDTree>(new NanoKDTree(3 /*dim*/, m_pointCloud, KDTreeSingleIndexAdaptorParams(10 /* max leaf */)));
			tree->buildIndex();
			return tree;

		}

        float m_regSqrt;

        CameraParameters m_camParamsHost;
		SimpleMesh m_result;

		PointCloud_nanoflann m_pointCloud;
		std::unique_ptr<NanoKDTree> m_targetAccelerationStructure;
		std::vector<float3> m_previousConstraints;

		SimpleMesh m_target;
		SimpleMesh m_averageMesh;
		Eigen::MatrixXf m_blendshapes;

		std::vector<unsigned int> m_dimsBasis;

        std::shared_ptr<ThalloImage> m_camParams;
        std::shared_ptr<ThalloImage> m_targetPos2;

		std::shared_ptr<ThalloImage> m_targetPos3;
		std::shared_ptr<ThalloImage> m_averageMeshPos3;
		std::shared_ptr<ThalloImage> m_blendshapeBasis;

		std::shared_ptr<ThalloImage> m_blendshapeWeights;
		std::shared_ptr<ThalloImage> m_vertexPos3;
};
