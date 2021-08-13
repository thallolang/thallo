#pragma once

typedef struct Thallo_State 	Thallo_State;
typedef struct Thallo_Plan 		Thallo_Plan;
typedef struct Thallo_Problem 	Thallo_Problem;

// Parameters that are set once per initialization of Thallo
// A zeroed-out version of this struct is a good default 
// for maximum speed on well-behaved problems
struct Thallo_InitializationParameters {
	// If true (nonzero), all intermediate values and unknowns, are double-precision
	// On platforms without double-precision float atomics, this 
	// can be a drastic drag of performance.
	int doublePrecision;

	// Valid Values: 0, no verbosity; 1, full verbosity
	int verbosityLevel;

	// Valid Values:
	//    0: No timing recorded
	//    1: Coarse-grained timing results
	//    2: Kernel-level timing info, using cuda events
	//    3: Invasive but accurate kernel-level timing by synchronizing before and after each call
	//           This will slow down the solver, but hopefully provide more accurate timing information
	int timingLevel;

	// Default block size for kernels (in threads). 
	// Must be a positive multiple of 32; if not, will default to 256.
	int threadsPerBlock;

	// Global flag for overriding all manual schedules with the autoscheduler
    int useAutoscheduler;

    // Valid Values: 0, regular CUDA backend; 1, experimental slow cpu backend
    int cpuOnly;
};

typedef struct Thallo_InitializationParameters 	Thallo_InitializationParameters;

// Allocate a new independant context for Thallo
Thallo_State* Thallo_NewState(Thallo_InitializationParameters params);

// load the problem specification including the energy function from 'filename' and
// initializer a solver of type 'solverkind' (currently only 'gauss_newton' 
// and 'levenberg_marquardt' are supported)
Thallo_Problem* Thallo_ProblemDefine(Thallo_State* state, const char* filename, const char* solverkind);
void Thallo_ProblemDelete(Thallo_State* state, Thallo_Problem* problem);


// Allocate intermediate arrays necessary to run 'problem' on the dimensions listed in 'dimensions'
// how the dimensions are used is based on the problem specification (see 'writing problem specifications')
Thallo_Plan* Thallo_ProblemPlan(Thallo_State* state, Thallo_Problem* problem, unsigned int* dimensions);
void Thallo_PlanFree(Thallo_State * state, Thallo_Plan* plan);

// Set a solver-specific variable by name. For now, these values are "locked-in" after ProblemInit()
// Consult the solver-specific documentation for valid values and names
void Thallo_SetSolverParameter(Thallo_State* state, Thallo_Plan* plan, const char* name, void* value);

// Get a solver-specific variable by name. For now, these values are "locked-in" after ProblemInit()
// Consult the solver-specific documentation for valid values and names
void Thallo_GetSolverParameter(Thallo_State* state, Thallo_Plan* plan, const char* name, void* value);

// Run the solver until completion using the plan 'plan'. 'problemparams' are the problem-specific inputs 
// and outputs that define the problem, including images, graphs, and problem paramaters
// (see 'writing problem specifications').
void Thallo_ProblemSolve(Thallo_State* state, Thallo_Plan* plan, void** problemparams);


// use these two functions to control the outer solver loop on your own. In between iterations,
// problem parameters can be inspected and updated.

// run just the initialization for a problem, but do not do any outer steps.
void Thallo_ProblemInit(Thallo_State* state, Thallo_Plan* plan, void** problemparams);
// perform one outer iteration of the solver loop and return to the user.
// a zero return value indicates that the solver is finished according to its parameters
int Thallo_ProblemStep(Thallo_State* state, Thallo_Plan* plan, void** problemparams);

// Return the result of the cost function evaluated on the current unknowns
// If the solver is initialized to not use double precision, the return value
// will be upconverted from a float before being returned
double Thallo_ProblemCurrentCost(Thallo_State* state, Thallo_Plan* plan);


// Total time can be obtained by multiplying count*meanMS
struct Thallo_PerformanceEntry {
    unsigned int count;
    double minMS;
    double maxMS;
    double meanMS;
    double stddevMS;
};
typedef struct Thallo_PerformanceEntry 	Thallo_PerformanceEntry;

struct Thallo_PerformanceSummary {
	// Performance Statistics for full solves
	Thallo_PerformanceEntry total;
	// Performance Statistics for individual nonlinear iterations,
	// This is broken up into three rough categories below
	Thallo_PerformanceEntry nonlinearIteration;
	Thallo_PerformanceEntry nonlinearSetup;
	Thallo_PerformanceEntry linearSolve;
	Thallo_PerformanceEntry nonlinearResolve;
};
typedef struct Thallo_PerformanceSummary 	Thallo_PerformanceSummary;

void Thallo_GetPerformanceSummary(Thallo_State* state, Thallo_Plan* plan, Thallo_PerformanceSummary* summary);