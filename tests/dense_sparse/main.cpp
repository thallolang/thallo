extern "C" {
#include "Thallo.h"
}
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define THALLO_DOUBLE_PRECISION 0

#if THALLO_DOUBLE_PRECISION
#define THALLO_FLOAT double
#define THALLO_FLOAT2 double2
#else
#define THALLO_FLOAT float
#define THALLO_FLOAT2 float2
#endif

double solve(int dataCount, int sparseCorr, int* startNodes, int* endNodes, THALLO_FLOAT* params, THALLO_FLOAT* data, std::string name) {
    Thallo_InitializationParameters param = {};
    param.doublePrecision = THALLO_DOUBLE_PRECISION;
    param.verbosityLevel = 2;
    param.timingLevel = 2;
    //param.threadsPerBlock = 512;
    Thallo_State* state = Thallo_NewState(param);
    // load the Thallo DSL file containing the cost description
    Thallo_Problem* problem = Thallo_ProblemDefine(state, name.c_str(), "gauss_newton");
    // describe the dimensions of the instance of the problem
    unsigned int dims[] = { dataCount, 1, sparseCorr };
    Thallo_Plan* plan = Thallo_ProblemPlan(state, problem, dims);
    // run the solver
    void* problem_data[] = { params, data, endNodes, startNodes };
    Thallo_ProblemSolve(state, plan, problem_data);
    double cost = Thallo_ProblemCurrentCost(state, plan);
    Thallo_PlanFree(state, plan);
    Thallo_ProblemDelete(state, problem);
    return cost;
}


int main(){
    auto sq = [](float x){ return x*x; };
    const int dim = 8192;
    THALLO_FLOAT2 generatorParams = { sq(100.0), 102.0 };
    std::vector<THALLO_FLOAT2> dataPoints(dim);
    THALLO_FLOAT a = sqrtf(generatorParams.x);
    THALLO_FLOAT b = generatorParams.y;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-50.0, 50.0);
    for (int i = 0; i < dataPoints.size(); ++i) {
        THALLO_FLOAT x = float(i)*2.0*3.141592653589 / dim;
        THALLO_FLOAT y = (a*cos(b*x) + b*sin(a*x));
        //y = a*x + b;
        // Add in noise
        //y += dis(gen);
        dataPoints[i].x = sqrtf(x);
        dataPoints[i].y = y;

    }
    
    THALLO_FLOAT2 unknownInit = { sq(99.7f), 101.6f };

    THALLO_FLOAT *d_data, *d_unknown;
    cudaMalloc(&d_data, dim*sizeof(THALLO_FLOAT2));
    cudaMalloc(&d_unknown, sizeof(THALLO_FLOAT2));
    cudaMemcpy(d_data, dataPoints.data(), dim*sizeof(THALLO_FLOAT2), cudaMemcpyHostToDevice);
    

    int numSparseCorr = 8192;
    int *d_startNodes, *d_endNodes;
    cudaMalloc(&d_startNodes, numSparseCorr*sizeof(int));
    cudaMalloc(&d_endNodes, numSparseCorr*sizeof(int));



    std::vector<int> endNodes;
    std::uniform_int_distribution<> idis(0, numSparseCorr - 1);
    for (int i = 0; i < numSparseCorr; ++i) { endNodes.push_back(idis(gen)); }
    cudaMemset(d_startNodes, 0, numSparseCorr*sizeof(int));
    cudaMemcpy(d_endNodes, endNodes.data(), numSparseCorr*sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_unknown, &unknownInit, sizeof(THALLO_FLOAT2), cudaMemcpyHostToDevice);
    double cost = solve(dim, numSparseCorr, d_startNodes, d_endNodes, d_unknown, d_data, "curveFitting_combined.t");


    THALLO_FLOAT2 unknownResult = {};
    cudaMemcpy(&unknownResult, d_unknown, sizeof(THALLO_FLOAT2), cudaMemcpyDeviceToHost);

    std::cout << "Init "    << sqrtf(unknownInit.x)     << ", " << unknownInit.y << std::endl;
    std::cout << "Result "  << sqrtf(unknownResult.x)   << ", " << unknownResult.y << std::endl;
    std::cout << "Goal "    << sqrtf(generatorParams.x) << ", " << generatorParams.y << std::endl;

    printf("\ndense_sparse %g\n", cost);

    cudaFree(d_data);
    cudaFree(d_unknown);
    cudaFree(d_startNodes);
    cudaFree(d_endNodes);
    return 0;
}