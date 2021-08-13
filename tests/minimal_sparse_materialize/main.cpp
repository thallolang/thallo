extern "C" {
#include "Thallo.h"
}
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../shared/stb_image_write.h"

double solveGraphLaplacian(int width, float* unknown, float* target, const int edgeCount, int* edgeStart, int* edgeEnd) {
    Thallo_InitializationParameters param = {};
    param.doublePrecision = 0;
    param.verbosityLevel = 2;
    param.timingLevel = 2;
    //param.threadsPerBlock = 512;
    Thallo_State* state = Thallo_NewState(param);
    // load the Thallo DSL file containing the cost description
    Thallo_Problem* problem = Thallo_ProblemDefine(state, "minimal_sparse_materialize.t", "gauss_newton");
    // describe the dimensions of the instance of the problem
    unsigned int dims[] = { width, edgeCount };
    Thallo_Plan* plan = Thallo_ProblemPlan(state, problem, dims);
    // run the solver
    void* problem_data[] = { unknown, target, edgeStart, edgeEnd };
    Thallo_ProblemSolve(state, plan, problem_data);
    double cost = Thallo_ProblemCurrentCost(state, plan);
    Thallo_PlanFree(state, plan);
    Thallo_ProblemDelete(state, problem);
    return cost;
}

void saveMonochromeImage(char const * filename, const int width, const int height, float* d_data) {
    float *data = new float[width*height];
    cudaMemcpy(data, d_data, width*height*sizeof(float), cudaMemcpyDeviceToHost);

    unsigned char* convertedData = new unsigned char[width*height];
    for (int i = 0; i < width*height; ++i) {
        convertedData[i] = (unsigned char)(data[i] * 255);
    }
    stbi_write_png(filename, width, height, 1, convertedData, width);

    delete[] data;
    delete[] convertedData;
}


int main(){
    const int dim = 512;
    float scratch[dim];
    srand(0xDEADBEEF);
    for (int i = 0; i < dim; ++i) {
        scratch[i] = (float)((double)rand() / (double)RAND_MAX);
    }
    float *target, *unknown;
    
    size_t fSize = dim*sizeof(float);
    cudaMalloc(&target, fSize);
    cudaMalloc(&unknown, fSize);
    cudaMemcpy(target, scratch, fSize, cudaMemcpyHostToDevice);
    cudaMemcpy(unknown, target, fSize, cudaMemcpyDeviceToDevice);
    
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

    double cost = solveGraphLaplacian(dim, unknown, target, edgeCount, d_edgeStart, d_edgeEnd);
    saveMonochromeImage("target.png", dim, 1, target);
    saveMonochromeImage("result.png", dim, 1, unknown);
    printf("\nminimal_fitting %g\n", cost);

    cudaFree(target);
    cudaFree(unknown);
    cudaFree(d_edgeStart);
    cudaFree(d_edgeEnd);
    return 0;
}