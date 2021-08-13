extern "C" {
#include "Thallo.h"
}
#include <cstdlib>
#include <iostream>
#include "../../examples/shared/cudaUtil.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../shared/stb_image_write.h"

double solveLaplacian(unsigned int width, unsigned int height, float* unknown, float* target, int* Xn, int* Yn) {
    Thallo_InitializationParameters param = {};
    param.doublePrecision = 0;
    param.verbosityLevel = 2;
    param.timingLevel = 2;
#ifdef THALLO_CPU
    param.cpuOnly = 1;
#endif
    //param.threadsPerBlock = 512;
    Thallo_State* state = Thallo_NewState(param);
    // load the Thallo DSL file containing the cost description
    Thallo_Problem* problem = Thallo_ProblemDefine(state, "laplacian.t", "gauss_newton");
    // describe the dimensions of the instance of the problem
    unsigned int dims[] = { width, height };
    Thallo_Plan* plan = Thallo_ProblemPlan(state, problem, dims);
    // run the solver
    void* problem_data[] = { unknown, target, Xn, Yn };
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
    float* scratch = new float[dim*dim];
    for (int i = 0; i < dim*dim; ++i) {
        scratch[i] = (float)((double)rand() / (double)RAND_MAX);
    }
    float *target, *unknown;
    size_t fSize = dim*dim*sizeof(float);
    cudaMalloc((void**)&target, fSize);
    cudaMalloc((void**)&unknown, fSize);
    cudaMemcpy(target, scratch, fSize, cudaMemcpyHostToDevice);
    cudaMemcpy(unknown, target, fSize, cudaMemcpyDeviceToDevice);
    delete[] scratch;

    int* Xn = new int[dim*dim];
    int* Yn = new int[dim*dim];


    for (int y = 0; y < dim; ++y) {
        bool yEdge = (y == (dim - 1));
        for (int x = 0; x < dim; ++x) {
            bool xEdge = (x == (dim - 1));
            int i = y*dim + x;
            Xn[i] = (xEdge) ? x : x + 1;
            Yn[i] = (yEdge) ? y : y + 1;
        }
    }

    int *Xn_d, *Yn_d;
    size_t iSize = dim*dim*sizeof(int);
    cudaMalloc((void**)&Xn_d, iSize);
    cudaMalloc((void**)&Yn_d, iSize);
    cudaMemcpy(Xn_d, Xn, iSize, cudaMemcpyHostToDevice);
    cudaMemcpy(Yn_d, Yn, iSize, cudaMemcpyHostToDevice);

    double cost = solveLaplacian(dim, dim, unknown, target, Xn_d,Yn_d);
    
    saveMonochromeImage("target.png", dim, dim, target);
    saveMonochromeImage("result.png", dim, dim, unknown);
    printf("\nminimal %g\n", cost);
    cudaFree(target);
    cudaFree(unknown);
    return 0;
}
