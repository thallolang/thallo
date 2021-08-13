#pragma once
extern "C" {
#include "Thallo.h"
}
#include "SolverIteration.h"
#include "NamedParameters.h"
#include <vector>
#include "cudaUtil.h"
#include <cmath>



inline void _assertM(const char* expression, std::string message, const char* file, int line)
{
    fprintf(stderr, "Assertion '%s' failed '%s:%d\n%s\n", expression, file, line, message.c_str());
    abort();
}

#define alwaysAssertM(EXPRESSION, MESSAGE) ((EXPRESSION) ? (void)0 : _assertM(#EXPRESSION, MESSAGE, __FILE__, __LINE__))

#ifdef _WIN32
#include <Windows.h>
class SimpleTimer {
public:
    void init() {
        // get ticks per second
        QueryPerformanceFrequency(&frequency);

        // start timer
        QueryPerformanceCounter(&lastTick);
    }
    // Time since last tick in ms
    double tick() {
        LARGE_INTEGER currentTick;
        QueryPerformanceCounter(&currentTick);

        // compute and print the elapsed time in millisec
        double elapsedTime = (currentTick.QuadPart - lastTick.QuadPart) * 1000.0 / frequency.QuadPart;
        lastTick = currentTick;
        return elapsedTime;
    }
protected:
    LARGE_INTEGER frequency;
    LARGE_INTEGER lastTick;
};
#else
#include <sys/time.h>

static double WallTimeInSeconds() {
  timeval time_val;
  gettimeofday(&time_val, NULL);
  return (time_val.tv_sec + time_val.tv_usec * 1e-6);
}
#include <chrono>
class SimpleTimer {
public:
    void init() {
      lastTick = WallTimeInSeconds();//std::chrono::high_resolution_clock::now();
    }
    // Time since last tick in ms
    double tick() {
      auto currentTick = WallTimeInSeconds();//std::chrono::high_resolution_clock::now();
      double elapsedTime = (currentTick-lastTick)*1000.0;//std::chrono::duration_cast<double, std::chrono::milliseconds>(currentTick - lastTick).count();
      lastTick = currentTick;
      return elapsedTime;
    }
 protected:
    //    std::chrono::time_point<std::chrono::high_resolution_clock> lastTick;
    double lastTick;
};
#endif



static void launchProfiledSolve(Thallo_State* state, Thallo_Plan* plan, void** problemParams, std::vector<SolverIteration>& iterationSummary) {
    SimpleTimer t;
    t.init();
    Thallo_ProblemInit(state, plan, problemParams);
    cudaDeviceSynchronize();
    double timeMS = t.tick();
    double cost = Thallo_ProblemCurrentCost(state, plan);
    iterationSummary.push_back(SolverIteration(cost, timeMS));

    t.tick();
    while (Thallo_ProblemStep(state, plan, problemParams)) {
        cudaDeviceSynchronize();
        timeMS = t.tick();
        cost = Thallo_ProblemCurrentCost(state, plan);
        iterationSummary.push_back(SolverIteration(cost, timeMS));
        t.tick();
    }
}


template<class T> size_t index_of(T element, const std::vector<T>& v) {
    auto location = std::find(v.begin(), v.end(), element);
    if (location != v.end()) {
        return std::distance(v.begin(), location);
    }
    else {
        return (size_t)-1;
    }
}

template<class T> T* getTypedParameterImage(std::string name, const NamedParameters& solverParameters) {
    auto i = index_of(name, solverParameters.names());
    alwaysAssertM(i != (size_t)-1, "Couldn't find parameter " + name);
    return (T*)(solverParameters.data()[i]);
}

static std::shared_ptr<ThalloGraph> getGraphFromParams(std::string name, const NamedParameters& solverParameters) {
    auto i = index_of(name, solverParameters.names());
    alwaysAssertM(i != (size_t)-1, "Couldn't find parameter " + name);
    NamedParameters::Parameter param;
    solverParameters.get(name, param);
    alwaysAssertM(param.graph != nullptr, "Graph was null!");
    return param.graph;
}
// TODO: Error handling
template<class T> void findAndCopyArrayToCPU(std::string name, std::vector<T>& cpuBuffer, const NamedParameters& solverParameters) {
    auto i = index_of(name, solverParameters.names());
    alwaysAssertM(i != (size_t)-1, "Couldn't find parameter " + name);
    cudaSafeCall(cudaMemcpy(cpuBuffer.data(), solverParameters.data()[i], sizeof(T)*cpuBuffer.size(), cudaMemcpyDeviceToHost));
}
template<class T> void findAndCopyToArrayFromCPU(std::string name, std::vector<T>& cpuBuffer, const NamedParameters& solverParameters) {
    auto i = index_of(name, solverParameters.names());
    alwaysAssertM(i != (size_t)-1, "Couldn't find parameter " + name);
    cudaSafeCall(cudaMemcpy(solverParameters.data()[i], cpuBuffer.data(), sizeof(T)*cpuBuffer.size(), cudaMemcpyHostToDevice));
}
template<class T> T getTypedParameter(std::string name, const NamedParameters& params) {
    auto i = index_of(name, params.names());
    alwaysAssertM(i != (size_t)-1, "Couldn't find parameter " + name);
    return *(T*)params.data()[i];
}

template<class T> void getTypedParameterIfPresent(std::string name, const NamedParameters& params, T& value) {
    auto i = index_of(name, params.names());
    if (i != (size_t)-1) {
        value = *(T*)params.data()[i];
    }
}


static void setAllSolverParameters(Thallo_State* state, Thallo_Plan* plan, const NamedParameters& params) {
    for (auto param : params.getVector()) {
        Thallo_SetSolverParameter(state, plan, param.name.c_str(), param.ptr);
    }
}

