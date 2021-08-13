#pragma once

#include "mLibInclude.h"
#ifndef THALLO_CPU
#include "CUDASolver/CUDASolverBundling.h"
#endif
#include "../../shared/cudaUtil.h"

#include "../../shared/SolverIteration.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/CombinedSolverBase.h"
#include "../../shared/ThalloGraph.h"

class CombinedSolver : public CombinedSolverBase
{
	public:
        CombinedSolver(CombinedSolverParameters combinedParams, SolverInput input, SolverState state, SolverParameters parameters) : CombinedSolverBase("Bundle Fusion Solve"),
            m_cudaSolverParams(parameters), m_intrinsics(input.intrinsics), m_imageWidth((float)input.denseDepthWidth), m_imageHeight((float)input.denseDepthHeight)
		{
            m_combinedSolverParameters = combinedParams;
            m_usesDense = parameters.useDense;

            float3 initialTrans0    = { 0.01f, 0.02f, 0.03f };
            float3 initialRot0      = { 0.011f, 0.000021f, 0.00031f };
            cudaSafeCall(cudaMemcpy(state.d_xTrans, &initialTrans0, sizeof(float3), cudaMemcpyHostToDevice));
            cudaSafeCall(cudaMemcpy(state.d_xRot, &initialRot0, sizeof(float3), cudaMemcpyHostToDevice));

            cudaTrans = state.d_xTrans;
            cudaRot = state.d_xRot;

            uint validCorrespondences = initializeSparseCorrespondenceValues(input.d_correspondences, input.numberOfCorrespondences);
            initializeUnknowns(state, input.numberOfImages);
            if (m_usesDense) {
                initializeDenseImPairConnectivity(input.numberOfImages, parameters.useDenseDepthAllPairwise);
                initializeImageStreams(input);
            }

            uint densePairCount = m_usesDense ? *(m_densePairGraph->edgeCountPtr()) : 0;
            std::vector<uint> dims = { input.denseDepthWidth, input.denseDepthHeight, input.numberOfImages, validCorrespondences, densePairCount };
            printf("Dense Pair Count = %u\n", densePairCount);

            free((void*)input.weightsSparse);
            free((void*)input.weightsDenseDepth);
            size_t weightSize = sizeof(float)*parameters.nNonLinearIterations;
            input.weightsSparse     = (const float*)malloc(weightSize);
            input.weightsDenseDepth = (const float*)malloc(weightSize);
            input.weightsDenseColor = (const float*)malloc(weightSize);
            memset((void*)input.weightsDenseColor, 0, weightSize);
            for (uint i = 0; i < parameters.nNonLinearIterations; ++i)  {
                float* ws = (float*)(input.weightsSparse);
                float* wd = (float*)(input.weightsDenseDepth);
                ws[i] = parameters.weightSparse;
                wd[i] = parameters.weightDenseDepth;
            }
            printf("# Linear iterations %u\n", parameters.nLinIterations);

            addThalloSolvers(dims);
#ifndef THALLO_CPU
            addSolver(std::make_shared<CUDASolverBundling>(input, state, parameters), "Cuda", m_combinedSolverParameters.useCUDA);
#endif
		} 

        void resetUnknowns() {
            
            int numImages = m_translation->dims()[0];

            copyImage(m_rotation, m_rotationOrig);
            copyImage(m_translation, m_translationOrig);
            std::vector<float3> h_translation, h_rotation;
            h_translation.resize(numImages);
            h_rotation.resize(numImages);
            cudaSafeCall(cudaMemcpy(h_translation.data(), m_translation->data(), m_translation->dataSize(), cudaMemcpyDeviceToHost));
            cudaSafeCall(cudaMemcpy(h_rotation.data(), m_rotation->data(), m_rotation->dataSize(), cudaMemcpyDeviceToHost));
            for (int i = 0; i < numImages; ++i) {
                auto t = h_translation[i];
                auto r = h_rotation[i];
            }
        }

        void initializeUnknowns(SolverState& state, uint numImages) {
            std::vector<uint> dims = { numImages };
            m_rotation = createEmptyThalloImage(dims, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, true);
            m_translation = createEmptyThalloImage(dims, ThalloImage::Type::FLOAT, 3, ThalloImage::GPU, true);
            printf("%p\n", m_translation->data());
            cudaSafeCall(cudaMemset(m_translation->data(), 0, sizeof(float) * 3 * numImages));
            cudaSafeCall(cudaMemset(m_rotation->data(), 0, sizeof(float) * 3 * numImages));
            cudaSafeCall(cudaMemcpy(m_translation->data(), state.d_xTrans, sizeof(float) * 3 * numImages, cudaMemcpyDeviceToDevice));
            cudaSafeCall(cudaMemcpy(m_rotation->data(), state.d_xRot, sizeof(float) * 3 * numImages, cudaMemcpyDeviceToDevice));
            m_rotationOrig = copyImageTo(m_rotation, ThalloImage::GPU);
            m_translationOrig = copyImageTo(m_translation, ThalloImage::GPU);
        }



        void initializeImageStreams(const SolverInput& input) {
            if (m_usesDense) {
                std::vector<CUDACachedFrame> h_cacheFrames;
                h_cacheFrames.resize(input.numberOfImages);
                cudaSafeCall(cudaMemcpy(h_cacheFrames.data(), input.d_cacheFrames, sizeof(CUDACachedFrame)*input.numberOfImages, cudaMemcpyDeviceToHost));
                std::vector<uint> dims = { input.denseDepthWidth, input.denseDepthHeight, input.numberOfImages };
                size_t pixCount = dims[0] * dims[1];
                size_t floatImSize = sizeof(float4) * pixCount;
                size_t ucharImSize = sizeof(uchar4) * pixCount;
                float4* d_positionData;
                float4* d_floatNormalData;
                uchar4* d_ucharNormalData;
                cudaSafeCall(cudaMalloc(&d_positionData, floatImSize * input.numberOfImages));
                cudaSafeCall(cudaMalloc(&d_floatNormalData, floatImSize * input.numberOfImages));
                cudaSafeCall(cudaMalloc(&d_ucharNormalData, ucharImSize * input.numberOfImages));
                for (uint i = 0; i < input.numberOfImages; ++i) {
                    cudaSafeCall(cudaMemcpy(d_positionData + (pixCount*i), h_cacheFrames[i].d_cameraposDownsampled, floatImSize, cudaMemcpyDeviceToDevice));
                    cudaSafeCall(cudaMemcpy(d_floatNormalData + (pixCount*i), h_cacheFrames[i].d_normalsDownsampled, floatImSize, cudaMemcpyDeviceToDevice));
                    cudaSafeCall(cudaMemcpy(d_ucharNormalData + (pixCount*i), h_cacheFrames[i].d_normalsDownsampledUCHAR4, ucharImSize, cudaMemcpyDeviceToDevice));
                }
                bool giveOwnership = true;
                bool isUnknown = false;
                m_positionImages = std::make_shared<ThalloImage>(dims, d_positionData, ThalloImage::FLOAT, 4, ThalloImage::GPU, isUnknown, giveOwnership);
                m_normalImages = std::make_shared<ThalloImage>(dims, d_floatNormalData, ThalloImage::FLOAT, 4, ThalloImage::GPU, isUnknown, giveOwnership);
                m_normalUcharImages = std::make_shared<ThalloImage>(dims, d_ucharNormalData, ThalloImage::UCHAR, 4, ThalloImage::GPU, isUnknown, giveOwnership);
            }
        }

        uint initializeSparseCorrespondenceValues(EntryJ* d_correspondences, uint numCorrespondences) {
            std::vector<EntryJ> h_correspondences;
            h_correspondences.resize(numCorrespondences);
            cudaSafeCall(cudaMemcpy(h_correspondences.data(), d_correspondences, numCorrespondences*sizeof(EntryJ), cudaMemcpyDeviceToHost));
            std::vector<std::vector<int>> thalloCorrespondences;
            std::vector<float3> cPosI;
            std::vector<float3> cPosJ;
            thalloCorrespondences.resize(2);

            std::sort(h_correspondences.begin(), h_correspondences.end(), [](EntryJ i, EntryJ j) {
                if (!i.isValid()) return false;
                if (!j.isValid()) return true;
                return (i.imgIdx_i == j.imgIdx_i && i.imgIdx_j < j.imgIdx_j) || i.imgIdx_i < j.imgIdx_i;
            });

            for (auto c : h_correspondences) {
                if (!c.isValid() || (c.imgIdx_i == c.imgIdx_j)) printf("Invalid correspondence!\n");
                else {
                    thalloCorrespondences[0].push_back(c.imgIdx_i);
                    thalloCorrespondences[1].push_back(c.imgIdx_j);
                    cPosI.push_back(c.pos_i);
                    cPosJ.push_back(c.pos_j);
                }
            }
            uint corrCount = (uint)cPosI.size();
            m_correspondenceGraph = std::make_shared<ThalloGraph>(thalloCorrespondences);
            m_correspondencePosI = copyImageTo(std::make_shared<ThalloImage>(std::vector<uint>({ corrCount }), cPosI.data(), ThalloImage::FLOAT, 3, ThalloImage::CPU), ThalloImage::GPU);
            m_correspondencePosJ = copyImageTo(std::make_shared<ThalloImage>(std::vector<uint>({ corrCount }), cPosJ.data(), ThalloImage::FLOAT, 3, ThalloImage::CPU), ThalloImage::GPU);
            return corrCount;
        }

        void initializeDenseImPairConnectivity(int numImages, bool useDenseDepthAllPairwise) {
            std::vector<std::vector<int>> imPairIndices;
            imPairIndices.resize(2);
            if (useDenseDepthAllPairwise) {
                for (int i = 0; i < numImages - 1; ++i) {
                    for (int j = i + 1; j < numImages; ++j) {
                        imPairIndices[0].push_back(i);
                        imPairIndices[1].push_back(j);
                    }
                }
            } else {
                for (int i = 0; i < numImages - 1; ++i) {
                    int j = i + 1;
                    imPairIndices[0].push_back(i);
                    imPairIndices[1].push_back(j);
                }
            }
            printf("ImPairIndices[0].size() == %lu\n", imPairIndices[0].size());
            printf("numImages == %d\n", numImages);
            m_densePairGraph = std::make_shared<ThalloGraph>(imPairIndices);
        }

        virtual void combinedSolveInit() override {
            m_problemParams.set("CamTranslation", m_translation);
            m_problemParams.set("CamRotation", m_rotation);
            if (m_usesDense) {
                m_problemParams.set("Positions", m_positionImages);
                m_problemParams.set("Normals", m_normalImages);
            }
            m_problemParams.set("Pos_j", m_correspondencePosJ);
            m_problemParams.set("Pos_i", m_correspondencePosI);
            if (m_usesDense) {
                m_problemParams.set("depthMin", &m_cudaSolverParams.denseDepthMin);
                m_problemParams.set("depthMax", &m_cudaSolverParams.denseDepthMax);
                m_problemParams.set("normalThresh", &m_cudaSolverParams.denseNormalThresh);
                m_problemParams.set("distThresh", &m_cudaSolverParams.denseDistThresh);
                m_problemParams.set("fx", &m_intrinsics.x);
                m_problemParams.set("fy", &m_intrinsics.y);
                m_problemParams.set("cx", &m_intrinsics.z);
                m_problemParams.set("cy", &m_intrinsics.w);
                m_problemParams.set("imageWidth", &m_imageWidth);
                m_problemParams.set("imageHeight", &m_imageHeight);
            }
            m_problemParams.set("weightSparse", &m_cudaSolverParams.weightSparse);
            if (m_usesDense) {
                m_problemParams.set("weightDenseDepth", &m_cudaSolverParams.weightDenseDepth);
            }
            m_problemParams.set("Correspondences", m_correspondenceGraph);
            if (m_usesDense) {
                m_problemParams.set("DensePairs", m_densePairGraph);
            }

            m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
            m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);
            
        }

        virtual void preNonlinearSolve(int i) override { }
        virtual void postNonlinearSolve(int) override{ 
            resetUnknowns(); 
        }

        virtual void preSingleSolve() override {
            
        }
        virtual void postSingleSolve() override { }
        virtual void combinedSolveFinalize() override {}

        ~CombinedSolver() {
        }

    private:
        float3* cudaTrans;
        float3* cudaRot;
        float m_imageWidth;
        float m_imageHeight;
        SolverParameters m_cudaSolverParams;
        float4 m_intrinsics;

        bool m_usesDense;

        std::shared_ptr<ThalloImage> m_translationOrig;
        std::shared_ptr<ThalloImage> m_rotationOrig;

        std::shared_ptr<ThalloImage> m_translation;
        std::shared_ptr<ThalloImage> m_rotation;

        std::shared_ptr<ThalloImage> m_positionImages;
        std::shared_ptr<ThalloImage> m_normalImages;
        std::shared_ptr<ThalloImage> m_normalUcharImages;

        std::shared_ptr<ThalloImage> m_correspondencePosI;
        std::shared_ptr<ThalloImage> m_correspondencePosJ;
        std::shared_ptr<ThalloGraph> m_correspondenceGraph;
        std::shared_ptr<ThalloGraph> m_densePairGraph;
};
