extern "C" {
#include "Thallo.h"
}
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <random>
#include <vector>

#define THALLO_DOUBLE_PRECISION 0

#if THALLO_DOUBLE_PRECISION
#define THALLO_FLOAT double
#else
#define THALLO_FLOAT float
#endif

double solve(int dataCount, THALLO_FLOAT* offset, THALLO_FLOAT* pts, THALLO_FLOAT* target, std::string name) {
    Thallo_InitializationParameters param = {};
    param.doublePrecision = THALLO_DOUBLE_PRECISION;
    param.verbosityLevel = 2;
    param.timingLevel = 2;
    //param.threadsPerBlock = 512;
    Thallo_State* state = Thallo_NewState(param);
    // load the Thallo DSL file containing the cost description
    Thallo_Problem* problem = Thallo_ProblemDefine(state, name.c_str(), "levenberg_marquardt");
    // describe the dimensions of the instance of the problem
    unsigned int dims[] = { dataCount, 1 };
    Thallo_Plan* plan = Thallo_ProblemPlan(state, problem, dims);
    // run the solver
    void* problem_data[] = { offset, pts, target };
    Thallo_ProblemSolve(state, plan, problem_data);
    double cost = Thallo_ProblemCurrentCost(state, plan);
    Thallo_PlanFree(state, plan);
    Thallo_ProblemDelete(state, problem);
    return cost;
}


int main(){

    const int dim = 512;
    THALLO_FLOAT generatorOffset = 10.0;
    std::vector<THALLO_FLOAT> dataPoints(dim);
    std::vector<THALLO_FLOAT> target(dim);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    THALLO_FLOAT sum = 0.0;
    for (int i = 0; i < dataPoints.size(); ++i) {
        dataPoints[i] = 0.0f;
        target[i] = generatorOffset + dis(gen);
        sum += target[i];
    }
    THALLO_FLOAT mean = sum / dataPoints.size();

    THALLO_FLOAT initOffset = 0.0;

    THALLO_FLOAT *d_offset, *d_pts, *d_target;
    cudaMalloc(&d_offset, sizeof(THALLO_FLOAT));
    cudaMalloc(&d_pts, dim*sizeof(THALLO_FLOAT));
    cudaMalloc(&d_target, dim*sizeof(THALLO_FLOAT));
    cudaMemcpy(d_offset, &initOffset, sizeof(THALLO_FLOAT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pts, dataPoints.data(), dim*sizeof(THALLO_FLOAT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target.data(), dim*sizeof(THALLO_FLOAT), cudaMemcpyHostToDevice);
    
    double cost = solve(dim, d_offset, d_pts, d_target, "multidomain.t");


    THALLO_FLOAT unknownOffset = 0.0;
    cudaMemcpy(&unknownOffset, d_offset, sizeof(THALLO_FLOAT), cudaMemcpyDeviceToHost);

    std::cout << std::setprecision(9) << "Init " << initOffset << std::endl;
    std::cout << "Result " << unknownOffset << std::endl;
    std::cout << "Goal   " << mean << ", " << generatorOffset << std::endl;
    printf("\nmultidomain %g\n", cost);
    cudaFree(d_offset);
    cudaFree(d_pts);
    cudaFree(d_target);
    return 0;
}