extern "C" {
#include "Thallo.h"
}
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../shared/stb_image_write.h"

#define UNKNOWN_CHANNELS 2


double solveGraphComplicated(int unknownCount, float* unknown, float* correspondences, const int edgeCount, int* edgeStart, int* edgeEnd) {
    Thallo_InitializationParameters param = {};
    param.doublePrecision = 0;
    param.verbosityLevel = 1;
    param.timingLevel = 2;
    //param.threadsPerBlock = 512;
    Thallo_State* state = Thallo_NewState(param);
    // load the Thallo DSL file containing the cost description
    Thallo_Problem* problem = Thallo_ProblemDefine(state, "complicated.t", "gauss_newton");
    // describe the dimensions of the instance of the problem
    unsigned int dims[] = { unknownCount, edgeCount };
    Thallo_Plan* plan = Thallo_ProblemPlan(state, problem, dims);
    // run the solver
    void* problem_data[] = { unknown, correspondences, edgeStart, edgeEnd };
    Thallo_ProblemSolve(state, plan, problem_data);
    double cost = Thallo_ProblemCurrentCost(state, plan);
    Thallo_PlanFree(state, plan);
    Thallo_ProblemDelete(state, problem);
    return cost;
}

void saveTriColorImage(char const * filename, const int width, const int height, float* d_data) {
    float *data = new float[width*height*3];
    cudaMemcpy(data, d_data, width*height*sizeof(float)*3, cudaMemcpyDeviceToHost);

    unsigned char* convertedData = new unsigned char[width*height*3];
    for (int i = 0; i < width*height*3; ++i) {
        convertedData[i] = (unsigned char)(data[i] * 255);
    }
    stbi_write_png(filename, width, height, 3, convertedData, width*3);

    delete[] data;
    delete[] convertedData;
}


int main(){
    const int dim = 512;
    const int unknownScalarCount = UNKNOWN_CHANNELS*dim;
    float scratch[unknownScalarCount];
    srand(0xdeadbeef);
    for (int i = 0; i < unknownScalarCount; ++i) {
        scratch[i] = (float)((double)rand() / (double)RAND_MAX);
        //scratch[i] = 0.0f;
    }
    float *target, *unknown, *original;
    size_t fSize = sizeof(float)*unknownScalarCount;
    cudaMalloc(&target, fSize);
    cudaMalloc(&unknown, fSize);
    cudaMalloc(&original, fSize);
    cudaMemcpy(target, scratch, fSize, cudaMemcpyHostToDevice);
    cudaMemcpy(unknown, target, fSize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(original, target, fSize, cudaMemcpyDeviceToDevice);
    
    const int edgeCount = dim - 1;
    int edgeStart[edgeCount] = {};
    int edgeEnd[edgeCount] = {};
    for (int i = 0; i < edgeCount; ++i) {
        edgeStart[i] = i;
        edgeEnd[i] = i + 1;
    }

    int *d_edgeStart, *d_edgeEnd;

    size_t edgeMemSize = edgeCount*sizeof(int);
    cudaMalloc(&d_edgeStart, edgeMemSize);
    cudaMalloc(&d_edgeEnd, edgeMemSize);
    cudaMemcpy(d_edgeStart, edgeStart, edgeMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeEnd, edgeEnd, edgeMemSize, cudaMemcpyHostToDevice);

    double cost = solveGraphComplicated(dim, unknown, target, edgeCount, d_edgeStart, d_edgeEnd);
    saveTriColorImage("original.png", dim*2, 1, original);
    saveTriColorImage("result.png", dim*2, 1, unknown);
    printf("\ncomplicated_graph_energy %g\n", cost);

    cudaFree(target);
    cudaFree(unknown);
    cudaFree(original);
    cudaFree(d_edgeStart);
    cudaFree(d_edgeEnd);
    return 0;
}