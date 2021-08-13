#include "CUDASolverBundling.h"
#include "../../../shared/CUDATimer.h"

extern "C" void solveBundlingStub(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverStateAnalysis& analysis, float* convergenceAnalysis, CUDATimer* timer, bool dumpInputOutput);
extern "C" float EvalResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer);

CUDASolverBundling::CUDASolverBundling(const SolverInput& input, const SolverState& state, const SolverParameters& parameters)
    : m_input(input), m_state(state), m_parameters(parameters), THREADS_PER_BLOCK(512) {
    auto maxNumDenseImPairs = input.maxNumberOfImages * (input.maxNumberOfImages - 1) / 2;
    auto maxNumResiduals = MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * maxNumDenseImPairs;
    unsigned int n = (maxNumResiduals + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cutilSafeCall(cudaMalloc(&m_solverExtra.d_maxResidual, sizeof(float) * n));
    cutilSafeCall(cudaMalloc(&m_solverExtra.d_maxResidualIndex, sizeof(int) * n));
    m_solverExtra.h_maxResidual = new float[n];
    m_solverExtra.h_maxResidualIndex = new int[n];
}

CUDASolverBundling::~CUDASolverBundling() {
    if (m_solverExtra.h_maxResidual) { free(m_solverExtra.h_maxResidual); }
    if (m_solverExtra.h_maxResidualIndex) { free(m_solverExtra.h_maxResidualIndex); }
    if (m_solverExtra.d_maxResidual) { cudaFree(m_solverExtra.d_maxResidual); }
    if (m_solverExtra.d_maxResidualIndex) { cudaFree(m_solverExtra.d_maxResidualIndex); }
}

double CUDASolverBundling::solve(const NamedParameters& solverParameters, const NamedParameters& problemParameters, SolverPerformanceSummary& stats, bool profileSolve, std::vector<SolverIteration>& iter)
{
    CUDATimer timer;
    m_finalCost = EvalResidual(m_input, m_state, m_parameters, &timer);
    printf("CUDA Initial Cost: %f\n", m_finalCost);
    solveBundlingStub(m_input, m_state, m_parameters, m_solverExtra, nullptr, &timer, false);
    m_finalCost = EvalResidual(m_input, m_state, m_parameters, &timer);
    cudaDeviceSynchronize();
    timer.evaluate(stats);
    m_summaryStats = stats;
    printf("CUDA Final Cost: %f\n", m_finalCost);
    return m_finalCost;
}
