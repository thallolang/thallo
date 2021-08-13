extern "C" {
#include "Thallo.h"
}
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#define _USE_MATH_DEFINES
#include <math.h>
#include "../shared/CudaArray.h"

double solveFitting(int numSamples, int convSize, float* convolution, float* original, float* target) {
    Thallo_InitializationParameters param = {};
    param.doublePrecision = 0;
    param.verbosityLevel = 2;
    param.timingLevel = 2;
    //param.threadsPerBlock = 512;
    Thallo_State* state = Thallo_NewState(param);
    // load the Thallo DSL file containing the cost description
    Thallo_Problem* problem = Thallo_ProblemDefine(state, "convolution.t", "gauss_newton");
    // describe the dimensions of the instance of the problem
    unsigned int dims[] = { numSamples, convSize };
    Thallo_Plan* plan = Thallo_ProblemPlan(state, problem, dims);

    unsigned nIterations = 1;
    Thallo_SetSolverParameter(state, plan, "nIterations", &nIterations);

    // run the solver
    void* problem_data[] = { convolution, original, target };
    Thallo_ProblemSolve(state, plan, problem_data);
    double cost = Thallo_ProblemCurrentCost(state, plan);
    Thallo_PlanFree(state, plan);
    Thallo_ProblemDelete(state, problem);
    return cost;
}

int main(){
    const int numSamples = 512;
    const int convSize = 5;
    std::vector<float> h_original, h_target, h_convolution, h_solutionConvolution;
    h_original.resize(numSamples);
    h_target.resize(numSamples);
    h_convolution = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    h_solutionConvolution = { -1.0f, -0.5f, 1.0f, -0.5f, 2.0f };

    CudaArray<float> convolution, original, target;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-50.0, 50.0);
    
    for (int i = 0; i < numSamples; ++i) {
        h_original[i] = (float)dis(gen);
        h_target[i] = h_original[i];
    }
    int off = convSize / 2;
    for (int i = off; i < numSamples - off; ++i) {
        float r = 0.0f;
        for (int c = 0; c < convSize; ++c) {
            r += h_original[i - c + off] * h_solutionConvolution[c];
        }
        h_target[i] = r;
    }

    convolution.update(h_convolution);
    original.update(h_original);
    target.update(h_target);

    double cost = solveFitting(numSamples, convSize, convolution.data(), original.data(), target.data());

    convolution.readBack(h_convolution);
    for (int i = 0; i < convSize; ++i) {
        printf("%d: %g = %f - %f\n", i, h_convolution[i] - h_solutionConvolution[i], h_convolution[i], h_solutionConvolution[i]);
    }
    printf("\nconvolution %g\n", cost);
    return 0;
}