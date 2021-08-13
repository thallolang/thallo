#include "CombinedSolver.h"
#include "bal_problem.h"

#include <tclap/CmdLine.h>
int main(int argc, const char * argv[]) {
    const std::string data_dir = "../data/";
    std::string filename = "../data/bal/problem-1778-993923-pre.txt";
    bool performanceProfiling = false;
    float max_solver_time_in_seconds = 600.0f;
    float eta = 0.1f;

    CombinedSolverParameters params;
    params.nonLinearIter = 5;
    params.linearIter = 150;
    //params.useThallo = false;
    //params.useThallo = true;
    params.useThalloLM = true;
    params.useCUDA = false;

    //params.thalloDoublePrecision = false;
    params.autoschedulerSetting = 1;
    params.thallofile = "bundle_adjustment.t";

    bool schedexp = false;
    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {
        TCLAP::CmdLine cmd("Bundle Adjustment", ' ', "1.0");
        TCLAP::SwitchArg perfSwitch("p", "perfprof", "Generate all data for performance profiles", cmd);
        TCLAP::SwitchArg scheduleExplorationSwitch("s", "schedexp", "Setup for a schedule exploration", cmd);
        TCLAP::ValueArg<std::string> thalloArg("o", "thallofile", "File to use for Thallo", false, params.thallofile, "string", cmd);
        TCLAP::ValueArg<int> autoArg("a", "autoschedule", "Autoschedule level", false, params.autoschedulerSetting, "int", cmd);
        TCLAP::ValueArg<float> timeArg("t", "timelimit", "Maximum Solver Time, in seconds", false, max_solver_time_in_seconds, "float", cmd);
        TCLAP::ValueArg<float> etaArg("e", "eta", "Eta, the constant forcing sequence for truncated linear solves in LM", false, eta, "float", cmd);
        TCLAP::UnlabeledValueArg<std::string> fileArg("file", "Filename", false, filename, "string", cmd);

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Get the value parsed by each arg.
        params.autoschedulerSetting = autoArg.getValue();
        performanceProfiling = perfSwitch.getValue();
        params.thallofile = thalloArg.getValue();
        schedexp = scheduleExplorationSwitch.getValue();
        filename = fileArg.getValue();
        max_solver_time_in_seconds = timeArg.getValue();
        eta = etaArg.getValue();
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    if (schedexp) {
        params.useThallo = true;
        params.useThalloLM = false;
        params.linearIter = 10;
    }

    std::string fileonly = filename.substr(filename.rfind("/") + 1);
    std::string noext = fileonly.substr(0, fileonly.find("."));

    std::shared_ptr<BALProblem> bal_problem = std::shared_ptr<BALProblem>(new BALProblem(filename, true));

    bal_problem->WriteToPLYFile(noext+".ply");

    CombinedSolver solver(bal_problem, params, noext, performanceProfiling, max_solver_time_in_seconds, eta);
    solver.solveAll();

	return 0;
}
