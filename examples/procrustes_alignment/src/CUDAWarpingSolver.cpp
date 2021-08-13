#include "CUDAWarpingSolver.h"
#include "../../shared/ThalloUtils.h"
extern "C" double ProcrustesSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverPerformanceSummary& stats);	// gauss newton

CUDAWarpingSolver::CUDAWarpingSolver(unsigned int N) : m_N(N)
{
	const unsigned int THREADS_PER_BLOCK = 512; // keep consistent with the GPU
	const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;
	const unsigned int numberOfVariables = 3; // This is different to all sparse solvers it is not N, we need a distinction between residuals and variables!

    m_solverInput.N = m_N;

	// State
	cudaSafeCall(cudaMalloc(&m_solverState.d_delta,				sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_deltaA,			sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_r,					sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_rA,				sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_z,					sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_zA,				sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_p,					sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_pA,				sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_Ap_X,				sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_Ap_A,				sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,			sizeof(float)*tmpBufferSize));
	cudaSafeCall(cudaMalloc(&m_solverState.d_scanBeta,			sizeof(float)*tmpBufferSize));
	cudaSafeCall(cudaMalloc(&m_solverState.d_rDotzOld,			sizeof(float)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_precondioner,		sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_precondionerA,		sizeof(float3)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_sumResidual,		sizeof(float)));
}

CUDAWarpingSolver::~CUDAWarpingSolver()
{
	// State
	cudaSafeCall(cudaFree(m_solverState.d_delta));
	cudaSafeCall(cudaFree(m_solverState.d_deltaA));
	cudaSafeCall(cudaFree(m_solverState.d_r));
	cudaSafeCall(cudaFree(m_solverState.d_rA));
	cudaSafeCall(cudaFree(m_solverState.d_z));
	cudaSafeCall(cudaFree(m_solverState.d_zA));
	cudaSafeCall(cudaFree(m_solverState.d_p));
	cudaSafeCall(cudaFree(m_solverState.d_pA));
	cudaSafeCall(cudaFree(m_solverState.d_Ap_X));
	cudaSafeCall(cudaFree(m_solverState.d_Ap_A));
	cudaSafeCall(cudaFree(m_solverState.d_scanAlpha));
	cudaSafeCall(cudaFree(m_solverState.d_scanBeta));
	cudaSafeCall(cudaFree(m_solverState.d_rDotzOld));
	cudaSafeCall(cudaFree(m_solverState.d_precondioner));
	cudaSafeCall(cudaFree(m_solverState.d_precondionerA));
	cudaSafeCall(cudaFree(m_solverState.d_sumResidual));
}

float sq(float x) { return x*x; }

double CUDAWarpingSolver::solve(const NamedParameters& solverParams, const NamedParameters& probParams, SolverPerformanceSummary& stats, bool profileSolve, std::vector<SolverIteration>& iters)
{
	
    m_solverState.d_mesh		= getTypedParameterImage<float3>("Mesh", probParams);
    m_solverState.d_target		= getTypedParameterImage<float3>("Target", probParams);
	m_solverState.d_translation = getTypedParameterImage<float3>("Translation", probParams);
	m_solverState.d_angles		= getTypedParameterImage<float3>("Angle", probParams);

    SolverParameters parameters;
    parameters.nNonLinearIterations = getTypedParameter<unsigned int>("nIterations", solverParams);
    parameters.nLinIterations       = getTypedParameter<unsigned int>("lIterations", solverParams);
    
    double finalCost = ProcrustesSolveGNStub(m_solverInput, m_solverState, parameters, stats);
    m_summaryStats = stats;
    return finalCost;
}
