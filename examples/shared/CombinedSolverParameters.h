#pragma once

struct CombinedSolverParameters {
    bool useCUDA = false;
    bool useThallo = true;
    bool useThalloLM = false;
    bool earlyOut = false;
    unsigned int numIter = 1;
    unsigned int nonLinearIter = 3;
    unsigned int linearIter = 200;
    unsigned int patchIter = 32;
    bool invasiveTiming = false;
    bool profileSolve = true;
    bool thalloDoublePrecision = false;
    int autoschedulerSetting = 1;
    std::string thallofile;
    bool thalloCPU = false;
};