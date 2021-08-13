#include "CombinedSolver.h"

static uint clamp(uint x, uint lo, uint hi) {
    return std::max(std::min(x, hi), lo);
}

static DepthImage32 loadDepthImage(std::string filename) {
    DepthImage32 image;
    FreeImageWrapper::loadImage(filename, image);
    return image;
}

#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {
    CombinedSolverParameters params;
    params.autoschedulerSetting = 1;
    params.thallofile = "deconvolution.t";

    params.invasiveTiming = false;
    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {
        TCLAP::CmdLine cmd("Thallo Example", ' ', "1.0");
        TCLAP::SwitchArg invasiveSwitch("i", "invasiveTiming", "Invasive Timing", cmd);
        TCLAP::ValueArg<int> autoArg("a", "autoschedule", "Autoschedule level", false, params.autoschedulerSetting, "int", cmd);
        TCLAP::ValueArg<std::string> thalloArg("o", "thallofile", "File to use for Thallo", false, params.thallofile, "string", cmd);

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Get the value parsed by each arg.
        params.autoschedulerSetting = autoArg.getValue();
        params.thallofile = thalloArg.getValue();
        params.invasiveTiming = invasiveSwitch.getValue();
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    std::string prefix = "../data/simple_proximal/";
    std::unordered_map<std::string, DepthImage32> inputs;

    auto loadInput = [&](std::string filename, std::string paramName){
        inputs[paramName] = loadDepthImage(prefix + filename);
    };

    loadInput("b1.tif", "b_1");
    loadInput("b2.tif", "b_2");
    loadInput("b3.tif", "b_3");
    loadInput("K.tif", "K");
    loadInput("dx.tif", "dx");
    loadInput("dy.tif", "dy");
    loadInput("M.tif", "M");
    loadInput("lambda.tif", "lambda");
    loadInput("x0.tif", "x0");
    DepthImage32 target = loadDepthImage(prefix + "result.tif");

    alwaysAssertM(inputs["lambda"].getDimX()*inputs["lambda"].getDimY() == 2, "lambda wrong size");

    auto dx = inputs["dx"];
    auto dy = inputs["dy"];
    printf("dx.getWidth() == %d\n", dx.getWidth());
    printf("dx.getHeight() == %d\n", dx.getHeight());
    printf("dy.getWidth() == %d\n", dy.getWidth());
    printf("dy.getHeight() == %d\n", dy.getHeight());
    alwaysAssertM(dx.getWidth() == 1 && dx.getHeight() == 2, "dx wrong size");
    alwaysAssertM(dx.getData()[0] == 1.0f && dx.getData()[1] == -1.0f, "dx wrong data");
    alwaysAssertM(dy.getWidth() == 2 && dy.getHeight() == 1, "dy wrong size");
    alwaysAssertM(dy.getData()[0] == 1.0f && dy.getData()[1] == -1.0f, "dy wrong data");

    params.numIter = 1;
    params.nonLinearIter = 1;
    params.linearIter = 3;
    params.useThallo = true;
    
    CombinedSolver solver(inputs, params);
    solver.solveAll();
    BaseImage<float> result = solver.result();


    double totalError = 0.0;
    double initialError = 0.0;
    auto x0 = inputs["x0"];
    auto b1 = inputs["b_1"];
    auto b3 = inputs["b_3"];

    auto get = [](BaseImage<float>& im, uint x, uint y) {
        return im(clamp(x, 0, im.getWidth()-1), clamp(y, 0, im.getHeight()-1));
    };

    BaseImage<float> diff = BaseImage<float>(result);
    BaseImage<float> Dxx = BaseImage<float>(result);
    BaseImage<float> Dyx = BaseImage<float>(result);
    ColorImageR8G8B8 v(result.getWidth(), result.getHeight());
    for (uint y = 0; y < result.getHeight(); ++y) {
        for (uint x = 0; x < result.getWidth(); ++x) {
            float f = result(x, y);
            float t = target(x, y);
            float error = ((f - t)*(f - t));
            totalError += (double)error;
            diff.setPixel(x, y, error);
            float o = x0(x, y);
            initialError += (double)((t - o)*(t - o));
            Dxx.setPixel(x, y, -x0(x, y) + get(x0, x + 1, y));
            Dyx.setPixel(x, y, (x0(x, y) - get(x0, x, y + 1)));
            v.setPixel(x, y, vec3f(f - t, t - f, 0)*255.0f);
        }
    }
    printf("Initial Error: %g\n", sqrt(initialError));
    printf("Total Error: %g\n", sqrt(totalError));
    FreeImageWrapper::saveImage("diff.tif", diff);
    FreeImageWrapper::saveImage("out.tif", result);
    FreeImageWrapper::saveImage("Dxx.tif", Dxx);
    FreeImageWrapper::saveImage("Dyx.tif", Dyx);
    FreeImageWrapper::saveImage("b3.tif", inputs["b_3"]);
    LodePNG::save(v, "v.png");

	return 0;
}
