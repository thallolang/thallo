#include "main.h"
#include "CUDASolver/SolverBundlingState.h"
#include "CombinedSolver.h"
#include <string>


#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {

    const std::string data_dir = "../data/";
    std::string filename = data_dir + "bundle_adjustment_dump_input.bin";

    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 1;
    params.thallofile = "bundle_fusion_solve.t";
    params.invasiveTiming = false;
    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {
        TCLAP::CmdLine cmd("Bundle Fusion", ' ', "1.0");
        TCLAP::SwitchArg perfSwitch("p", "perf", "Performance Run", cmd);
        TCLAP::SwitchArg invasiveSwitch("i", "invasiveTiming", "Invasive Timing", cmd);
        TCLAP::ValueArg<int> autoArg("a", "autoschedule", "Autoschedule level", false, params.autoschedulerSetting, "int", cmd);
        TCLAP::ValueArg<std::string> thalloArg("o", "thallofile", "File to use for Thallo", false, params.thallofile, "string", cmd);
        TCLAP::UnlabeledValueArg<std::string> fileArg("file", "Filename", false, filename, "string", cmd);

        // Parse the argv array.
        cmd.parse(argc, argv);


        // Get the value parsed by each arg.
        performanceRun = perfSwitch.getValue();
        params.autoschedulerSetting = autoArg.getValue();
        params.thallofile = thalloArg.getValue();
        params.invasiveTiming = invasiveSwitch.getValue();
        filename = fileArg.getValue();
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    
    SolverState state;
    SolverInput input;
    SolverParameters bundlerParams;
    loadAllStateFromFile(filename, input, state, bundlerParams, true, 1);

    params.numIter = 1;
    params.useThallo = true;

    // For better timing
    bundlerParams.nNonLinearIterations = std::min((uint)3, bundlerParams.nNonLinearIterations);
    params.numIter = 5;
    params.useCUDA = performanceRun;
    params.profileSolve = false;

    printf("input.numberOfImages*3 = %d\n", input.numberOfImages * 3);
    printf("bundlerParams.nLinIterations = %d\n", bundlerParams.nLinIterations);
    printf("bundlerParams.nNonLinearIterations = %d\n", bundlerParams.nNonLinearIterations);
    if (bundlerParams.useDense) {
        bundlerParams.nLinIterations = std::min(input.numberOfImages*3, bundlerParams.nLinIterations);
    } else {
        bundlerParams.nLinIterations = (bundlerParams.nLinIterations / 5);
    }

    params.nonLinearIter = bundlerParams.nNonLinearIterations; 
    params.linearIter = bundlerParams.nLinIterations;
    printf("params.linearIter = %d\n", params.linearIter);
    params.thalloDoublePrecision = false;

    CombinedSolver solver(params, input, state, bundlerParams);
    solver.solveAll();


    return 0;
}
