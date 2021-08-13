#include "mLibInclude.h"
#include "CombinedSolver.h"
#include "ImageHelper.h"


void renderFlowVecotors(ColorImageR8G8B8A8& image, const BaseImage<float2>& flowVectors) {
	const unsigned int skip = 5;	//only every n-th pixel
	const float lengthRed = 5.0f;
	
	for (unsigned int j = 1; j < image.getHeight() - 1; j += skip) {
		for (unsigned int i = 1; i < image.getWidth() - 1; i += skip) {
			
			const float2& flowVector = flowVectors(i, j);
			vec2i start = vec2i(i, j);
			vec2i end = start + vec2i(math::round(flowVector.x), math::round(flowVector.y));
			float len = vec2f(flowVector.x, flowVector.y).length();
			vec4uc color = math::round(255.0f*BaseImageHelper::convertDepthToRGBA(len, 0.0f, 5.0f)*2.0f);	color.w = 255;
			//vec4uc color = math::round(255.0f*vec4f(0.1f, 0.8f, 0.1f, 1.0f));	//TODO color-code length

			ImageHelper::drawLine(image, start, end, color);
		}
	}
}

#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {
    std::string srcFile = "../data/dogdance0.png";
    std::string tarFile = "../data/dogdance1.png";

    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 1;
    params.thallofile = "optical_flow.t";
    params.invasiveTiming = false;
    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {
        TCLAP::CmdLine cmd("Thallo Example", ' ', "1.0");
        TCLAP::SwitchArg perfSwitch("p", "perf", "Performance Run", cmd);
        TCLAP::SwitchArg invasiveSwitch("i", "invasiveTiming", "Invasive Timing", cmd);
        TCLAP::ValueArg<int> autoArg("a", "autoschedule", "Autoschedule level", false, params.autoschedulerSetting, "int", cmd);
        TCLAP::ValueArg<std::string> thalloArg("o", "thallofile", "File to use for Thallo", false, params.thallofile, "string", cmd);
        TCLAP::UnlabeledMultiArg<std::string> inputArg("srctgt", "Source Target", false, "string", cmd);

        // Parse the argv array.
        cmd.parse(argc, argv);


        // Get the value parsed by each arg.
        performanceRun = perfSwitch.getValue();
        params.autoschedulerSetting = autoArg.getValue();
        params.thallofile = thalloArg.getValue();
        params.invasiveTiming = invasiveSwitch.getValue();
        auto inputs = inputArg.getValue();
        if (inputs.size() > 0) {
            assert(inputs.size() == 2);
            srcFile = inputs[0];
            tarFile = inputs[1];
        }
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

	ColorImageR8G8B8A8 imageSrc = LodePNG::load(srcFile);
	ColorImageR8G8B8A8 imageTar = LodePNG::load(tarFile);

	ColorImageR32 imageSrcGray = imageSrc.convertToGrayscale();
	ColorImageR32 imageTarGray = imageTar.convertToGrayscale();

    params.numIter = 3;
    params.nonLinearIter = 1;
    params.linearIter = 50;

    CombinedSolver solver(imageSrcGray, imageTarGray, params);
    solver.solveAll();
    BaseImage<float2> flowVectors = solver.result();

	const std::string outFile = "out.png";
	ColorImageR8G8B8A8 out = imageSrc;
	renderFlowVecotors(out, flowVectors);
	LodePNG::save(out, outFile);

	return 0;
}
