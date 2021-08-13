#include "main.h"
#include "CombinedSolver.h"
#include "SFSSolverInput.h"

#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {
    std::string inputFilenamePrefix = "../data/shape_from_shading/default";

    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 0;
    params.thallofile = "shape_and_shading.t";

    params.invasiveTiming = false;
    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {
        TCLAP::CmdLine cmd("Thallo Example", ' ', "1.0");
        TCLAP::SwitchArg perfSwitch("p", "perf", "Performance Run", cmd);
        TCLAP::SwitchArg invasiveSwitch("i", "invasiveTiming", "Invasive Timing", cmd);
        TCLAP::ValueArg<int> autoArg("a", "autoschedule", "Autoschedule level", false, params.autoschedulerSetting, "int", cmd);
        TCLAP::ValueArg<std::string> thalloArg("o", "thallofile", "File to use for Thallo", false, params.thallofile, "string", cmd);
        TCLAP::UnlabeledValueArg<std::string> inputFilenamePrefixArg("file", "Filename", false, inputFilenamePrefix, "string", cmd);

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Get the value parsed by each arg.
        performanceRun = perfSwitch.getValue();
        params.autoschedulerSetting = autoArg.getValue();
        params.thallofile = thalloArg.getValue();
        params.invasiveTiming = invasiveSwitch.getValue();
        inputFilenamePrefix = inputFilenamePrefixArg.getValue();
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    SFSSolverInput solverInputGPU;
    solverInputGPU.load(inputFilenamePrefix, true);
    solverInputGPU.targetDepth->savePLYMesh("sfsInitDepth.ply");

    float f_x = solverInputGPU.parameters.fx;
    float f_y = solverInputGPU.parameters.fy;
    float u_x = solverInputGPU.parameters.ux;
    float u_y = solverInputGPU.parameters.uy;
    int W = solverInputGPU.initialUnknown->width();
    SimpleBuffer intensity(*solverInputGPU.targetIntensity, false);

    auto isValid = [W](int x, int y, float* depthPtr) {
        float value = depthPtr[y*W + x];
        return (value > 0.01f && value <= 10000.0f);
    };
    auto saveFnOfDepth = [&intensity, &isValid](std::string filename, float* depthPtr, std::function<float(int, int, float*)> pixelFn) {
        std::vector<float> result;
        result.resize(intensity.height()*intensity.width());
        for (int y = 0; y < intensity.height(); ++y) {
            for (int x = 0; x < intensity.width(); ++x) {
                int i = y*intensity.width() + x;
                result[i] = 0.5f;
                if (isValid(x, y, depthPtr)) {
                    if (isValid(x - 1, y, depthPtr) && isValid(x, y - 1, depthPtr)) {
                        result[i] = pixelFn(x, y, depthPtr);
                    }
                }
            }
        }
        auto resultIm = SimpleBuffer((float*)result.data(), intensity.width(), intensity.height(), false);
        resultIm.savePNG(filename, 255.0f);
    };
    //solverInputCPU.load(inputFilenamePrefix, false);
    auto computeLighting = [](float3 n, float* ell) {
        return ell[0] +
            ell[1] * n.y + ell[2] * n.z + ell[3] * n.x +
            ell[4] * n.x*n.y + ell[5] * n.y*n.z + ell[6] * (-n.x*n.x - n.y*n.y + 2 * n.z*n.z) + ell[7] * n.z*n.x + ell[8] * (n.x*n.x - n.y*n.y);
    };

    auto normal = [f_x, f_y, u_x, u_y, W](float _x, float _y, float* depthPtr) {
        auto D_r = [W, depthPtr](int x, int y) {
            return depthPtr[y*W + x];
        };
        float n_x = D_r(_x, _y - 1) * (D_r(_x, _y) - D_r(_x - 1, _y)) / f_y;
        float n_y = D_r(_x - 1, _y) * (D_r(_x, _y) - D_r(_x, _y - 1)) / f_x;
        float n_z = (n_x * (u_x - _x) / f_x) + (n_y * (u_y - _y) / f_y) - (D_r(_x - 1, _y)*D_r(_x, _y - 1) / (f_x*f_y));
        float sqLength = n_x*n_x + n_y*n_y + n_z*n_z;
        float inverseMagnitude = (sqLength > 0.0) ? 1.0 / sqrt(sqLength) : 1.0;
        return float3{ inverseMagnitude*n_x, inverseMagnitude*n_y, inverseMagnitude*n_z };
    };

    float* ell_orig = solverInputGPU.parameters.lightingCoefficients;
    params.nonLinearIter = 27;
    params.linearIter = 18;
    //params.useThalloLM = true;
    if (performanceRun) {
        params.useThallo   = true;
        params.useThalloLM = true;
        params.nonLinearIter = 27;
        params.linearIter = 18;
    }

    params.useThallo = true;

    //params.useThalloLM = true;
    CombinedSolver solver(solverInputGPU, params);
    printf("Solving\n");
    solver.solveAll();
    std::shared_ptr<SimpleBuffer> result = solver.result();
    printf("Solved\n");
    printf("About to save\n");
    result->save("sfsOutput.imagedump");
    result->savePNG("sfsOutput", 150.0f);
    result->savePLYMesh("sfsOutput.ply");
    solverInputGPU.targetIntensity->savePNG("sfsInitIntensity", 255.0f);

    SimpleBuffer depthCPU(*result, false);
    float* depthPtrResult = (float*)depthCPU.data();
    SimpleBuffer depthCPUInit(*solverInputGPU.targetDepth, false);
    float* depthPtrInit = (float*)depthCPUInit.data();

    auto resultLighting = solver.resultLighting();
    float* ell_result = (float*)resultLighting->data();

    saveFnOfDepth("n_x", depthPtrResult, [&normal](int x, int y, float* depthPtr) {return normal(x, y, depthPtr).x*0.5f + 0.5f; });
    saveFnOfDepth("n_y", depthPtrResult, [&normal](int x, int y, float* depthPtr) {return normal(x, y, depthPtr).y*0.5f + 0.5f; });
    saveFnOfDepth("n_z", depthPtrResult, [&normal](int x, int y, float* depthPtr) {return normal(x, y, depthPtr).z*0.5f + 0.5f; });
    
    auto origLightingFn = [&computeLighting, &normal, ell_orig](int x, int y, float* depthPtr) {return computeLighting(normal(x, y, depthPtr), ell_orig); };
    auto resultLightingFn = [&computeLighting, &normal, ell_result](int x, int y, float* depthPtr) {return computeLighting(normal(x, y, depthPtr), ell_result); };

    saveFnOfDepth("result_w_orig_lighting", depthPtrResult, origLightingFn);
    saveFnOfDepth("result_w_result_lighting", depthPtrResult, resultLightingFn);
    saveFnOfDepth("init_w_orig_lighting", depthPtrInit, origLightingFn);
    saveFnOfDepth("init_w_result_lighting", depthPtrInit, resultLightingFn);


    printf("Save\n");

	return 0;
}
