#pragma once
// "Coincidentally" identical layout to Thallo version of this data.
struct SolverPerformanceEntry {
    unsigned int count = 0;
    double minMS = 0.0;
    double maxMS = 0.0;
    double meanMS = 0.0;
    double stddevMS = 0.0;
    SolverPerformanceEntry() {}
    SolverPerformanceEntry(double t) :count(1), minMS(t), maxMS(t), meanMS(t), stddevMS(0.0){}
};

struct SolverPerformanceSummary {
    // Performance Statistics for full solves
    SolverPerformanceEntry total;
    // Performance Statistics for individual nonlinear iterations,
    // This is broken up into three rough categories below
    SolverPerformanceEntry nonlinearIteration;
    SolverPerformanceEntry nonlinearSetup;
    SolverPerformanceEntry linearSolve;
    SolverPerformanceEntry nonlinearResolve;
    
};