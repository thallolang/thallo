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

double solveFitting(int numSamples, int numWeights, float* unknown, float* tmplt, float* target) {
    Thallo_InitializationParameters param = {};
    param.doublePrecision = 0;
    param.verbosityLevel = 2;
    param.timingLevel = 2;
    //param.threadsPerBlock = 512;
    Thallo_State* state = Thallo_NewState(param);
    // load the Thallo DSL file containing the cost description
    Thallo_Problem* problem = Thallo_ProblemDefine(state, "minimal_fitting.t", "gauss_newton");
    // describe the dimensions of the instance of the problem
    unsigned int dims[] = { numSamples, numWeights };
    Thallo_Plan* plan = Thallo_ProblemPlan(state, problem, dims);
    // run the solver
    void* problem_data[] = { unknown, tmplt, target };
    Thallo_ProblemSolve(state, plan, problem_data);
    double cost = Thallo_ProblemCurrentCost(state, plan);
    Thallo_PlanFree(state, plan);
    Thallo_ProblemDelete(state, problem);
    return cost;
}

int main(){
    const int numSamples = 512;
    const int numWeights = 16;
    std::vector<float> h_tmplt, h_target, h_unknown;
    h_tmplt.resize(numSamples*numWeights);
    h_target.resize(numSamples);
    h_unknown.resize(numWeights);

    CudaArray<float> target, unknown, tmplt;

    auto normalize = [numSamples](int i) {
        return (float(i)/(numSamples-1));
    };
    // Triangle wave with period of 1, defined on the range 0 to 1
    auto triangleWave = [](float x) {
        if (x < 0.25) {
            return 4.0f*x;
        } else if (x > 0.75) {
            return 4.0f*x - 4.0f;
        } else {
            return (-4.0f)*x + 2.0f;
        }
    };

    for (int i = 0; i < numSamples; ++i) {
        h_target[i] = triangleWave(normalize(i));
    }
    for (int m = 0; m < numWeights; ++m) {
        int n = m * 2 + 1;
        float predicted = (8.0f / (M_PI*M_PI)) / (n*n);
        predicted = (m % 2 == 0) ? predicted : -predicted;
        h_unknown[m] = 0.0f;//predicted;
    }
    for (int m = 0; m < numWeights; ++m) {
        for (int i = 0; i < numSamples; ++i) {
            int n = m * 2 + 1; // triangle wave is an odd function
            // Fourier series coefficients are 8/pi^2 (-1)^m/n^2 (http://mathworld.wolfram.com/FourierSeriesTriangleWave.html)
            h_tmplt[m*numSamples + i] = sinf((float)M_PI*n*normalize(i)*2.0f);
        }
    }
    for (int n = 0; n < numSamples; ++n) {
        float result = 0.0f;
        for (int m = 0; m < numWeights; ++m) {
            result += h_tmplt[m*numSamples + n] * h_unknown[m];
        }
        float off = result - h_target[n];
        float error = off*off;
        //("%d: %g, (target %g) offset %g\n", n, result, h_target[n], off);
    }


    tmplt.update(h_tmplt);
    target.update(h_target);
    unknown.update(h_unknown);

    double cost = solveFitting(numSamples, numWeights, unknown.data(), tmplt.data(), target.data());
    unknown.readBack(h_unknown);
    float totalError = 0.0f;

    for (int m = 0; m < numWeights; ++m) {
        int n = m * 2 + 1;
        float predicted = (8.0f / (M_PI*M_PI)) / (n*n);
        predicted = (m % 2 == 0) ? predicted : -predicted;
        printf("Result weight %d: %g (%g predicted)\n", m, h_unknown[m], predicted);
    }

    for (int n = 0; n < numSamples; ++n) {
        float result = 0.0f;
        for (int m = 0; m < numWeights; ++m) {
            result += h_tmplt[m*numSamples+n]*h_unknown[m];
        }
        float off = result-h_target[n];
        float error = off*off;
        totalError += error;
        //printf("%d: %g, offset %g\n", n, result, off);
        
    }
    printf("Total Error: %f\n", totalError);
    printf("\nminimal_fitting %g\n", cost);

    return 0;
}