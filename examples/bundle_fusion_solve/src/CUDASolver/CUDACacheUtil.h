#pragma once
#ifndef CUDA_CACHE_UTIL
#define CUDA_CACHE_UTIL

#include <cutil_inline.h>
#include <cutil_math.h>
#define HAS_CUTIL

#define CUDA_SAFE_FREE(b) { if(!b) { cutilSafeCall(cudaFree(b)); b = NULL; } }


#define MINF __int_as_float(0xff800000)

#define CUDACACHE_UCHAR_NORMALS
#define CUDACACHE_FLOAT_NORMALS

struct CUDACachedFrame {
	void alloc(unsigned int width, unsigned int height) {
        cutilSafeCall(cudaMalloc(&d_depthDownsampled, sizeof(float) * width * height));
		//MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_colorDownsampled, sizeof(uchar4) * width * height));
        cutilSafeCall(cudaMalloc(&d_cameraposDownsampled, sizeof(float4) * width * height));

        cutilSafeCall(cudaMalloc(&d_intensityDownsampled, sizeof(float) * width * height));
        cutilSafeCall(cudaMalloc(&d_intensityDerivsDownsampled, sizeof(float2) * width * height));
#ifdef CUDACACHE_UCHAR_NORMALS
        cutilSafeCall(cudaMalloc(&d_normalsDownsampledUCHAR4, sizeof(uchar4) * width * height));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
        cutilSafeCall(cudaMalloc(&d_normalsDownsampled, sizeof(float4) * width * height));
#endif
	}
	void free() {
        CUDA_SAFE_FREE(d_depthDownsampled);
		//MLIB_CUDA_SAFE_FREE(d_colorDownsampled);
        CUDA_SAFE_FREE(d_cameraposDownsampled);

        CUDA_SAFE_FREE(d_intensityDownsampled);
        CUDA_SAFE_FREE(d_intensityDerivsDownsampled);
#ifdef CUDACACHE_UCHAR_NORMALS
        CUDA_SAFE_FREE(d_normalsDownsampledUCHAR4);
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
        CUDA_SAFE_FREE(d_normalsDownsampled);
#endif
	}

	float* d_depthDownsampled;
	//uchar4* d_colorDownsampled;
	float4* d_cameraposDownsampled;

	//for dense color term
	float* d_intensityDownsampled; //this could be packed with intensityDerivaties to a float4 dunno about the read there
	float2* d_intensityDerivsDownsampled; //TODO could have energy over intensity gradient instead of intensity
#ifdef CUDACACHE_UCHAR_NORMALS
	uchar4* d_normalsDownsampledUCHAR4;
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
	float4* d_normalsDownsampled;
#endif
};

#endif //CUDA_CACHE_UTIL