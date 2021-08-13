#include "CUDAWarpingSolver.h"
#include "../../shared/ThalloUtils.h"
extern "C" double ImageWarpingSolveGNStub(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverPerformanceSummary& stats);	// gauss newton

CUDAWarpingSolver::CUDAWarpingSolver(unsigned int N, unsigned int M) : m_N(N), m_M(M)
{
	const unsigned int THREADS_PER_BLOCK = 512; // keep consistent with the GPU
	const unsigned int tmpBufferSize = THREADS_PER_BLOCK*THREADS_PER_BLOCK;
	const unsigned int numberOfVariables = M; // This is different to all sparse solvers it is not N, we need a distinction between residuals and variables!
	const unsigned int numberOfVertices = N;

    m_solverInput.N = m_N;
	m_solverInput.M = m_M;

	// State
	cudaSafeCall(cudaMalloc(&m_solverState.d_delta,				sizeof(float)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_r,					sizeof(float)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_z,					sizeof(float)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_p,					sizeof(float)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_Ap,				sizeof(float)*numberOfVariables));

	cudaSafeCall(cudaMalloc(&m_solverState.d_JTv, sizeof(float)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_Jv,  sizeof(float3)*numberOfVertices));
	cudaSafeCall(cudaMalloc(&m_solverState.d_Jv2,  sizeof(float)*numberOfVariables));

	cudaSafeCall(cudaMalloc(&m_solverState.d_scanAlpha,			sizeof(float)*tmpBufferSize));
	cudaSafeCall(cudaMalloc(&m_solverState.d_scanBeta,			sizeof(float)*tmpBufferSize));
	cudaSafeCall(cudaMalloc(&m_solverState.d_rDotzOld,			sizeof(float)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_precondioner,		sizeof(float)*numberOfVariables));
	cudaSafeCall(cudaMalloc(&m_solverState.d_sumResidual,		sizeof(float)));
}

CUDAWarpingSolver::~CUDAWarpingSolver()
{
	// State
	cudaSafeCall(cudaFree(m_solverState.d_delta));
	cudaSafeCall(cudaFree(m_solverState.d_r));
	cudaSafeCall(cudaFree(m_solverState.d_z));
	cudaSafeCall(cudaFree(m_solverState.d_p));
	cudaSafeCall(cudaFree(m_solverState.d_Ap));

	cudaSafeCall(cudaFree(m_solverState.d_JTv));
	cudaSafeCall(cudaFree(m_solverState.d_Jv));
	cudaSafeCall(cudaFree(m_solverState.d_Jv2));

	cudaSafeCall(cudaFree(m_solverState.d_scanAlpha));
	cudaSafeCall(cudaFree(m_solverState.d_scanBeta));
	cudaSafeCall(cudaFree(m_solverState.d_rDotzOld));
	cudaSafeCall(cudaFree(m_solverState.d_precondioner));
	cudaSafeCall(cudaFree(m_solverState.d_sumResidual));
}

float sq(float x) { return x*x; }

double CUDAWarpingSolver::solve(const NamedParameters& solverParams, const NamedParameters& probParams, SolverPerformanceSummary& perfStats, bool profileSolve, std::vector<SolverIteration>& iters)
{
    m_solverState.d_mesh		= getTypedParameterImage<float3>("Mesh", probParams);
    m_solverState.d_target		= getTypedParameterImage<float2>("Target", probParams);
	m_solverState.d_average		= getTypedParameterImage<float3>("AverageMesh", probParams);
    m_solverState.d_camParams   = getTypedParameterImage<float>("CamParams", probParams);

	m_solverState.d_blendshapeWeights = getTypedParameterImage<float>("BlendshapeWeights", probParams);
	m_solverState.d_blendshapeBasis   = getTypedParameterImage<float3>("BlendshapeBasis", probParams);

    m_solverInput.wReg = sq(getTypedParameter<float>("w_regSqrt", probParams));

    SolverParameters parameters;
    parameters.nNonLinearIterations = getTypedParameter<unsigned int>("nIterations", solverParams);
    parameters.nLinIterations       = getTypedParameter<unsigned int>("lIterations", solverParams);
    
	double finalCost = ImageWarpingSolveGNStub(m_solverInput, m_solverState, parameters, perfStats);
    m_summaryStats = perfStats;
    return finalCost;
}
