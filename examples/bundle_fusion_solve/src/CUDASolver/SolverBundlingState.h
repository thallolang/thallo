#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include <cuda_runtime.h> 
#include <vector>
#include "EntryJ.h"
#include "CUDACacheUtil.h"
#include "SolverBundlingParameters.h"
#include "cuda_SimpleMatrixUtil.h"
#include "GlobalDefines.h"
#include <random>
#include <sstream>

struct SolverInput
{
    EntryJ* d_correspondences;
    int* d_variablesToCorrespondences;
    int* d_numEntriesPerRow;

    unsigned int numberOfCorrespondences;
    unsigned int numberOfImages;

    unsigned int maxNumberOfImages;
    unsigned int maxCorrPerImage;

    const int* d_validImages;
    const CUDACachedFrame* d_cacheFrames;
    unsigned int denseDepthWidth;
    unsigned int denseDepthHeight;
    float4 intrinsics;				//TODO constant buffer for this + siftimagemanger stuff?
    unsigned int maxNumDenseImPairs;
    float2 colorFocalLength; //color camera params (actually same as depthIntrinsics...)

    const float* weightsSparse;
    const float* weightsDenseDepth;
    const float* weightsDenseColor;
};

// State of the GN Solver
struct SolverState
{
    float3*	d_deltaRot;					// Current linear update to be computed
    float3*	d_deltaTrans;				// Current linear update to be computed

    float3* d_xRot;						// Current state
    float3* d_xTrans;					// Current state

    float3*	d_rRot;						// Residuum // jtf
    float3*	d_rTrans;					// Residuum // jtf

    float3*	d_zRot;						// Preconditioned residuum
    float3*	d_zTrans;					// Preconditioned residuum

    float3*	d_pRot;						// Decent direction
    float3*	d_pTrans;					// Decent direction

    float3*	d_Jp;						// Cache values after J

    float3*	d_Ap_XRot;					// Cache values for next kernel call after A = J^T x J x p
    float3*	d_Ap_XTrans;				// Cache values for next kernel call after A = J^T x J x p

    float*	d_scanAlpha;				// Tmp memory for alpha scan

    float*	d_rDotzOld;					// Old nominator (denominator) of alpha (beta)

    float3*	d_precondionerRot;			// Preconditioner for linear system
    float3*	d_precondionerTrans;		// Preconditioner for linear system

    float*	d_sumResidual;				// sum of the squared residuals //debug

    //float* d_residuals; // debugging
    //float* d_sumLinResidual; // debugging // helper to compute linear residual

    int* d_countHighResidual;

    __host__ float getSumResidual() const {
        float residual;
        cudaMemcpy(&residual, d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
        return residual;
    }

    // for dense depth term
    float* d_denseJtJ;
    float* d_denseJtr;
    float* d_denseCorrCounts;

    float4x4* d_xTransforms;
    float4x4* d_xTransformInverses;

    uint2* d_denseOverlappingImages;
    int* d_numDenseOverlappingImages;

    //!!!DEBUGGING
    int* d_corrCount;
    int* d_corrCountColor;
    float* d_sumResidualColor;
};

struct SolverStateAnalysis
{
    // residual pruning
    int*	d_maxResidualIndex;
    float*	d_maxResidual;

    int*	h_maxResidualIndex;
    float*	h_maxResidual;
};

template <typename T>
void writePOD(std::ofstream& s, const T  v) {
    s.write((const char*)&v, sizeof(T));
}

template <typename T>
void writeArray(std::ofstream& s, const T* v, const uint count) {
    s.write((const char*)&count, sizeof(uint));
    s.write((const char*)v, sizeof(T)*count);
}

template <typename T>
void writeCudaArray(std::ofstream& s, const T* d_v, const uint count) {
    std::vector<T> vec;
    vec.resize(count);
    cutilSafeCall(cudaMemcpy(vec.data(), d_v, sizeof(T)*count, cudaMemcpyDeviceToHost));
    writeArray<T>(s, vec.data(), count);
}

template <typename T>
void readPOD(std::ifstream& s, T* v) {
    s.read((char*)v, sizeof(T));
}

template <typename T>
void readArray(std::ifstream& s, T** v, uint* count, bool allocate = false) {
    s.read((char*)count, sizeof(uint));
    if (allocate) { *v = (T*)malloc(sizeof(T)*(*count)); }
    s.read((char*)*v, sizeof(T)*(*count));
}

template <typename T>
void readCudaArray(std::ifstream& s, T** d_v, uint* count, bool allocate = false, int multiplyProblemSize = 1) {
    std::vector<T> vec;
    s.read((char*)count, sizeof(uint));
    vec.resize((*count)*multiplyProblemSize);
    size_t size = sizeof(T)*(*count);
    //printf("%u*%lu = %lu\n", *count, (unsigned long)sizeof(T), (unsigned long)size);
    s.read((char*)vec.data(), size);
    if (multiplyProblemSize > 1) {
        for (int j = 0; j < (*count); ++j) {
            for (int i = 1; i < multiplyProblemSize; ++i) {
                //printf("%d/%d\n", (*count)*i + j, vec.size());
                vec[(*count)*i + j] = vec[j];
            }
        }
    }
    
    if (allocate) {
        cutilSafeCall(cudaMalloc(d_v, size*multiplyProblemSize));
        //printf("%p\n", *d_v);
    }
    cutilSafeCall(cudaMemcpy(*d_v, vec.data(), size*multiplyProblemSize, cudaMemcpyHostToDevice));
    (*count) = (*count)*multiplyProblemSize;
}

template <typename T>
void duplicateCudaArray(T** d_dst, T* d_src, uint* count, bool allocate = false) {
    size_t size = sizeof(T)*(*count);
    if (allocate) {
        cutilSafeCall(cudaMalloc(d_dst, size));
    }
    cutilSafeCall(cudaMemcpy(*d_dst, d_src, size, cudaMemcpyDeviceToDevice));
}

struct SolverInputPOD {
    uint numCorr;
    uint numIm;
    uint maxIm;
    uint maxCorr;
    uint denseW;
    uint denseH;
    float4 intrinsics;
    uint maxNumDenseImPairs;
    float2 colorFocalLength;
    SolverInputPOD() {}
    SolverInputPOD(const SolverInput& input) {
        numCorr = input.numberOfCorrespondences;
        numIm = input.numberOfImages;
        maxIm = input.maxNumberOfImages;
        maxCorr = input.maxCorrPerImage;
        denseW = input.denseDepthWidth;
        denseH = input.denseDepthHeight;
        intrinsics = input.intrinsics;
        maxNumDenseImPairs = input.maxNumDenseImPairs;
        colorFocalLength = input.colorFocalLength;
    }
    void transferValues(SolverInput& input) {
        input.numberOfCorrespondences = numCorr;
        input.numberOfImages = numIm;
        input.maxNumberOfImages = maxIm;
        input.maxCorrPerImage = maxCorr;
        input.denseDepthWidth = denseW;
        input.denseDepthHeight = denseH;
        input.intrinsics = intrinsics;
        input.maxNumDenseImPairs = maxNumDenseImPairs;
        input.colorFocalLength = colorFocalLength;
    }
};

static void saveAllStateToFile(std::string filename, const SolverInput& input, const SolverState& state, const SolverParameters& parameters) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cout << "Error opening " << filename << " for write" << std::endl;
        return;
    }
    writePOD<SolverParameters>(out, parameters);
    SolverInputPOD inputPOD(input);
    writePOD<SolverInputPOD>(out, inputPOD);
    writeCudaArray<EntryJ>(out, input.d_correspondences, inputPOD.numCorr);
    writeCudaArray<int>(out, input.d_variablesToCorrespondences, inputPOD.numIm*inputPOD.maxCorr);
    writeCudaArray<int>(out, input.d_numEntriesPerRow, inputPOD.numIm);
    writeCudaArray<int>(out, input.d_validImages, inputPOD.numIm);
    int hasCache = input.d_cacheFrames ? 1 : 0;
    writePOD<int>(out, hasCache);
    if (hasCache) {
        std::vector<CUDACachedFrame> cacheFrames;
        cacheFrames.resize(inputPOD.numIm);
        uint numPix = input.denseDepthWidth*input.denseDepthHeight;
        printf("%p\n", input.d_cacheFrames);
        cutilSafeCall(cudaMemcpy(cacheFrames.data(), input.d_cacheFrames, sizeof(CUDACachedFrame)*inputPOD.numIm, cudaMemcpyDeviceToHost));
        for (auto f : cacheFrames) {
            writeCudaArray<float>(out, f.d_depthDownsampled, numPix);
            writeCudaArray<float4>(out, f.d_cameraposDownsampled, numPix);
            writeCudaArray<float>(out, f.d_intensityDownsampled, numPix); //this could be packed with intensityDerivaties to a float4 dunno about the read there
            writeCudaArray<float2>(out, f.d_intensityDerivsDownsampled, numPix); //TODO could have energy over intensity gradient instead of intensity
            writeCudaArray<float4>(out, f.d_normalsDownsampled, numPix);
            writeCudaArray<uchar4>(out, f.d_normalsDownsampledUCHAR4, numPix);
        }
    }
    writeArray<float>(out, input.weightsSparse, parameters.nNonLinearIterations);
    writeArray<float>(out, input.weightsDenseDepth, parameters.nNonLinearIterations);
    writeArray<float>(out, input.weightsDenseColor, parameters.nNonLinearIterations);
    writeCudaArray<float3>(out, state.d_xRot, inputPOD.numIm);
    writeCudaArray<float3>(out, state.d_xTrans, inputPOD.numIm);

    out.close();
}

static void allocateSolverState(SolverState& state, unsigned int numberOfVariables, unsigned int maxNumResiduals, unsigned int maxNumDenseImPairs) {
    // State
    cutilSafeCall(cudaMalloc(&state.d_deltaRot, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_deltaTrans, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_rRot, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_rTrans, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_zRot, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_zTrans, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_pRot, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_pTrans, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_Jp, sizeof(float3)*maxNumResiduals));
    cutilSafeCall(cudaMalloc(&state.d_Ap_XRot, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_Ap_XTrans, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_scanAlpha, sizeof(float) * 2));
    cutilSafeCall(cudaMalloc(&state.d_rDotzOld, sizeof(float) *numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_precondionerRot, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_precondionerTrans, sizeof(float3)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_sumResidual, sizeof(float)));
    cutilSafeCall(cudaMalloc(&state.d_countHighResidual, sizeof(int)));
    if (maxNumDenseImPairs > 0) {
        printf("maxNumDenseImPairs %u\n", maxNumDenseImPairs);
        cutilSafeCall(cudaMalloc(&state.d_denseJtJ, sizeof(float) * 36 * numberOfVariables * numberOfVariables));
        cutilSafeCall(cudaMalloc(&state.d_denseJtr, sizeof(float) * 6 * numberOfVariables));
        cutilSafeCall(cudaMalloc(&state.d_denseCorrCounts, sizeof(float) * maxNumDenseImPairs));
        cutilSafeCall(cudaMalloc(&state.d_denseOverlappingImages, sizeof(uint2) * maxNumDenseImPairs));
        cutilSafeCall(cudaMalloc(&state.d_numDenseOverlappingImages, sizeof(int)));
    }
    cutilSafeCall(cudaMalloc(&state.d_corrCount, sizeof(int)));
    cutilSafeCall(cudaMalloc(&state.d_corrCountColor, sizeof(int)));
    cutilSafeCall(cudaMalloc(&state.d_sumResidualColor, sizeof(float)));

#ifdef USE_LIE_SPACE
    cutilSafeCall(cudaMalloc(&state.d_xTransforms, sizeof(float4x4)*numberOfVariables));
    cutilSafeCall(cudaMalloc(&state.d_xTransformInverses, sizeof(float4x4)*numberOfVariables));
#else
    state.d_xTransforms = NULL;
    state.d_xTransformInverses = NULL;
#endif
}

static void loadAllStateFromFile(std::string filename, SolverInput& input, SolverState& state, SolverParameters& parameters, bool allocate = false, int multiplyProblemSize = 1) {
    std::ifstream inp(filename, std::ios::binary);
    if (!inp.is_open()) {
        std::cout << "Error opening " << filename << " for read" << std::endl;
        return;
    }
    readPOD<SolverParameters>(inp, &parameters);
    SolverInputPOD inputPOD;
    readPOD<SolverInputPOD>(inp, &inputPOD);
    inputPOD.transferValues(input);

    inputPOD.numIm = inputPOD.numIm*multiplyProblemSize;

    readCudaArray<EntryJ>(inp, &input.d_correspondences, &inputPOD.numCorr, allocate, multiplyProblemSize);
    uint numPix = input.denseDepthWidth*input.denseDepthHeight;
    uint variableToCorrCount = inputPOD.numIm*inputPOD.maxCorr;
    readCudaArray<int>(inp, &input.d_variablesToCorrespondences, &variableToCorrCount, allocate, multiplyProblemSize);
    readCudaArray<int>(inp, &input.d_numEntriesPerRow, &inputPOD.numIm, allocate, multiplyProblemSize);
    readCudaArray<int>(inp, ((int**)&input.d_validImages), &inputPOD.numIm, allocate, multiplyProblemSize);

    int hasCache = 0;
    readPOD<int>(inp, &hasCache);
    if (hasCache) {
        std::vector<CUDACachedFrame> cacheFrames;
        cacheFrames.resize(inputPOD.numIm);
        if (!allocate) { cutilSafeCall(cudaMemcpy(cacheFrames.data(), (void*)input.d_cacheFrames, sizeof(CUDACachedFrame)*inputPOD.numIm, cudaMemcpyDeviceToHost)); }
        for (int i = 0; i < cacheFrames.size() / multiplyProblemSize; ++i) {
            auto& f = cacheFrames[i];
            readCudaArray<float>(inp, &f.d_depthDownsampled, &numPix, allocate);
            readCudaArray<float4>(inp, &f.d_cameraposDownsampled, &numPix, allocate);
            readCudaArray<float>(inp, &f.d_intensityDownsampled, &numPix, allocate);
            readCudaArray<float2>(inp, &f.d_intensityDerivsDownsampled, &numPix, allocate);
            readCudaArray<float4>(inp, &f.d_normalsDownsampled, &numPix, allocate);
            readCudaArray<uchar4>(inp, &f.d_normalsDownsampledUCHAR4, &numPix, allocate);
            if (multiplyProblemSize > 1) {
                for (int j = 1; j < multiplyProblemSize; ++j) {
                    auto& f2 = cacheFrames[j*(cacheFrames.size() / multiplyProblemSize) + i];
                    duplicateCudaArray<float>(   &f2.d_depthDownsampled,           f.d_depthDownsampled,           &numPix, allocate);
                    duplicateCudaArray<float4>(  &f2.d_cameraposDownsampled,       f.d_cameraposDownsampled,       &numPix, allocate);
                    duplicateCudaArray<float>(   &f2.d_intensityDownsampled,       f.d_intensityDownsampled,       &numPix, allocate);
                    duplicateCudaArray<float2>(  &f2.d_intensityDerivsDownsampled, f.d_intensityDerivsDownsampled, &numPix, allocate);
                    duplicateCudaArray<float4>(  &f2.d_normalsDownsampled,         f.d_normalsDownsampled,         &numPix, allocate);
                    duplicateCudaArray<uchar4>(  &f2.d_normalsDownsampledUCHAR4,   f.d_normalsDownsampledUCHAR4,   &numPix, allocate);
                }
            }
        }
        if (allocate) { cudaMalloc(&input.d_cacheFrames, sizeof(CUDACachedFrame)*inputPOD.numIm); }
        cutilSafeCall(cudaMemcpy((void*)input.d_cacheFrames, cacheFrames.data(), sizeof(CUDACachedFrame)*inputPOD.numIm, cudaMemcpyHostToDevice));
    }
    else {
        input.d_cacheFrames = nullptr;
    }
    readArray<float>(inp, (float**)&input.weightsSparse, &parameters.nNonLinearIterations, allocate);
    readArray<float>(inp, (float**)&input.weightsDenseDepth, &parameters.nNonLinearIterations, allocate);
    readArray<float>(inp, (float**)&input.weightsDenseColor, &parameters.nNonLinearIterations, allocate);
    input.maxNumberOfImages *= multiplyProblemSize;
    printf("MaxNumberOfImages %d\n", input.maxNumberOfImages);
    if (allocate) {
        auto maxImPairs = (input.maxNumberOfImages * (input.maxNumberOfImages - 1) / 2);
        auto maxNumDenseImPairs = (input.d_cacheFrames) ? maxImPairs : 0;
        auto maxNumResiduals = MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * maxImPairs;
        maxNumResiduals = std::min((unsigned int)3432000, maxNumResiduals);//TODO:Remove
        allocateSolverState(state, input.maxNumberOfImages, maxNumResiduals, maxNumDenseImPairs);
    }
    readCudaArray<float3>(inp, &state.d_xRot, &inputPOD.numIm, allocate, multiplyProblemSize);
    readCudaArray<float3>(inp, &state.d_xTrans, &inputPOD.numIm, allocate, multiplyProblemSize);
    inp.close();
    inputPOD.transferValues(input);
}

template<typename T>
void printSample(std::string name, const T* ptr, int size, std::mt19937& gen, bool cuda = false, int count = -1) {
    if (count == -1) {
        count = clamp(size / 4, 1, 5);
    }
    if (cuda) {
        const T* h_ptr = (T*)malloc(size*sizeof(T));
        cutilSafeCall(cudaMemcpy((void*)h_ptr, (const void*)ptr, size*sizeof(T), cudaMemcpyDeviceToHost));
        ptr = h_ptr;
    }
    std::cout << name << ": {";
    std::uniform_int_distribution<> dis(0, size - 1);
    for (int i = 0; i < count; ++i) {
        int ind = dis(gen);
        std::cout << ind << ": " << ptr[ind];
        if (i < count - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "}" << std::endl;
    if (cuda) {
        free((void*)ptr);
    }
}

static std::ostream& operator<<(std::ostream& os, const float2& v) {
    os << "(" << v.x << "," << v.y << ")";
    return os;
}
static std::ostream& operator<<(std::ostream& os, const float3& v) {
    os << "(" << v.x << "," << v.y << "," << v.z << ")";
    return os;
}
static std::ostream& operator<<(std::ostream& os, const float4& v) {
    os << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")";
    return os;
}

static std::ostream& operator<<(std::ostream& os, const uchar4& v) {
    os << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")";
    return os;
}

static void printStochasticSubsetOfSolverValues(SolverInput& input, SolverState& state, SolverParameters& parameters) {
    std::mt19937 gen(0xdeadbeef);
    std::cout << "nNonLinearIterations" << parameters.nNonLinearIterations << std::endl;
    std::cout << "nLinIterations" << parameters.nLinIterations << std::endl;
    std::cout << "verifyOptDistThresh" << parameters.verifyOptDistThresh << std::endl;
    std::cout << "verifyOptPercentThresh" << parameters.verifyOptPercentThresh << std::endl;
    std::cout << "highResidualThresh" << parameters.highResidualThresh << std::endl;
    std::cout << "denseDistThresh" << parameters.denseDistThresh << std::endl;
    std::cout << "denseNormalThresh" << parameters.denseNormalThresh << std::endl;
    std::cout << "denseColorThresh" << parameters.denseColorThresh << std::endl;
    std::cout << "denseColorGradientMin" << parameters.denseColorGradientMin << std::endl;
    std::cout << "denseDepthMin" << parameters.denseDepthMin << std::endl;
    std::cout << "denseDepthMax" << parameters.denseDepthMax << std::endl;
    std::cout << "useDenseDepthAllPairwise" << parameters.useDenseDepthAllPairwise << std::endl;
    std::cout << "denseOverlapCheckSubsampleFactor" << parameters.denseOverlapCheckSubsampleFactor << std::endl;
    std::cout << "weightSparse" << parameters.weightSparse << std::endl;
    std::cout << "weightDenseDepth" << parameters.weightDenseDepth << std::endl;
    std::cout << "weightDenseColor" << parameters.weightDenseColor << std::endl;
    std::cout << "useDense" << parameters.useDense << std::endl;

    std::cout << "Num Corr: " << input.numberOfCorrespondences << std::endl;
    std::cout << "Num Im: " << input.numberOfImages << std::endl;
    std::cout << "Max Im: " << input.maxNumberOfImages << std::endl;
    std::cout << "Max Corr/Im: " << input.maxCorrPerImage << std::endl;
    std::cout << "DenseW: " << input.denseDepthWidth << std::endl;
    std::cout << "DenseH: " << input.denseDepthHeight << std::endl;
    std::cout << "Intrinsics: " << input.intrinsics.x << ", " << input.intrinsics.y << ", " << input.intrinsics.z << ", " << input.intrinsics.w << std::endl;
    std::cout << "maxNumDenseImPairs: " << input.maxNumDenseImPairs << std::endl;
    std::cout << "colorFocalLength: " << input.colorFocalLength.x << ", " << input.colorFocalLength.y << std::endl;

    std::cout << "d_correspondences: " << "NYI" << std::endl;
    printSample<int>("d_variablesToCorrespondences", input.d_variablesToCorrespondences, input.numberOfImages*input.maxCorrPerImage, gen, true);
    printSample<int>("d_numEntriesPerRow", input.d_numEntriesPerRow, input.numberOfImages, gen, true);
    printSample<int>("d_validImages", input.d_validImages, input.numberOfImages, gen, true);
    int hasCache = input.d_cacheFrames ? 1 : 0;
    if (hasCache) {
        uint numPix = input.denseDepthWidth*input.denseDepthHeight;
        std::vector<CUDACachedFrame> cacheFrames;
        cacheFrames.resize(input.numberOfImages);
        cutilSafeCall(cudaMemcpy(cacheFrames.data(), input.d_cacheFrames, sizeof(CUDACachedFrame)*input.numberOfImages, cudaMemcpyDeviceToHost));
        for (int i = 0; i < cacheFrames.size(); ++i) {
            std::cout << "CacheFrame " << i << ":" << std::endl;
            auto f = cacheFrames[i];
            printSample<float>("    d_depthDownsampled", f.d_depthDownsampled, numPix, gen, true);
            printSample<float4>("    d_cameraposDownsampled", f.d_cameraposDownsampled, numPix, gen, true);
            printSample<float>("    d_intensityDownsampled", f.d_intensityDownsampled, numPix, gen, true);
            printSample<float2>("    d_intensityDerivsDownsampled", f.d_intensityDerivsDownsampled, numPix, gen, true);
            printSample<float4>("    d_normalsDownsampled", f.d_normalsDownsampled, numPix, gen, true);
            printSample<uchar4>("    d_normalsDownsampledUCHAR4", f.d_normalsDownsampledUCHAR4, numPix, gen, true);
        }
    }
    printSample<float>("weightsSparse", input.weightsSparse, parameters.nNonLinearIterations, gen);
    printSample<float>("weightsDenseDepth", input.weightsDenseDepth, parameters.nNonLinearIterations, gen);
    printSample<float>("weightsDenseColor", input.weightsDenseColor, parameters.nNonLinearIterations, gen);

    printSample<float3>("d_xRot", state.d_xRot, input.numberOfImages, gen, true);
    printSample<float3>("d_xTrans", state.d_xTrans, input.numberOfImages, gen, true);
}

#endif
