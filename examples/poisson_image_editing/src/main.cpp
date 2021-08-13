#include "main.h"
#include "CombinedSolver.h"

#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {
    std::string inputImage0 = "../data/poisson0.png";
    std::string inputImage1 = "../data/poisson1.png";
    std::string inputImageMask = "../data/poisson_mask.png";

    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 1;
    params.thallofile = "poisson_image_editing.t";

    params.invasiveTiming = false;
    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {
        TCLAP::CmdLine cmd("Thallo Example", ' ', "1.0");
        TCLAP::SwitchArg perfSwitch("p", "perf", "Performance Run", cmd);
        TCLAP::SwitchArg invasiveSwitch("i", "invasiveTiming", "Invasive Timing", cmd);
        TCLAP::ValueArg<int> autoArg("a", "autoschedule", "Autoschedule level", false, params.autoschedulerSetting, "int", cmd);
        TCLAP::ValueArg<std::string> thalloArg("o", "thallofile", "File to use for Thallo", false, params.thallofile, "string", cmd);
        TCLAP::UnlabeledMultiArg<std::string> inputArg("inputs", "im0 im1 mask", false, "multi string", cmd);

        // Parse the argv array.
        cmd.parse(argc, argv);

        // Get the value parsed by each arg.
        performanceRun = perfSwitch.getValue();
        params.autoschedulerSetting = autoArg.getValue();
        params.thallofile = thalloArg.getValue();
        params.invasiveTiming = invasiveSwitch.getValue();
        auto inputs = inputArg.getValue();
        if (inputs.size() > 0) {
            assert(inputs.size() == 3);
            inputImage0 = inputs[0];
            inputImage1 = inputs[1];
            inputImageMask = inputs[2];
        }
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
	const unsigned int offsetX = 0;
	const unsigned int offsetY = 0;
	const bool invertMask = false;

    ColorImageR8G8B8A8	   image = LodePNG::load(inputImage0);
	ColorImageR32G32B32A32 imageR32(image.getWidth(), image.getHeight());
	for (unsigned int y = 0; y < image.getHeight(); y++) {
		for (unsigned int x = 0; x < image.getWidth(); x++) {
			imageR32(x,y) = image(x,y);
		}
	}

	ColorImageR8G8B8A8	   image1 = LodePNG::load(inputImage1);
	ColorImageR32G32B32A32 imageR321(image1.getWidth(), image1.getHeight());
	for (unsigned int y = 0; y < image1.getHeight(); y++) {
		for (unsigned int x = 0; x < image1.getWidth(); x++) {
			imageR321(x, y) = image1(x, y);
		}
	}

	ColorImageR32G32B32A32 image1Large = imageR32;
	image1Large.setPixels(ml::vec4uc(0, 0, 0, 255));
	for (unsigned int y = 0; y < imageR321.getHeight(); y++) {
		for (unsigned int x = 0; x < imageR321.getWidth(); x++) {
			image1Large(x + offsetY, y + offsetX) = imageR321(x, y);
		}
	}


	
	const ColorImageR8G8B8A8 imageMask = LodePNG::load(inputImageMask);
	ColorImageR32 imageR32Mask(imageMask.getWidth(), imageMask.getHeight());
	for (unsigned int y = 0; y < imageMask.getHeight(); y++) {
		for (unsigned int x = 0; x < imageMask.getWidth(); x++) {
			unsigned char c = imageMask(x, y).x;
			if (invertMask) {
				if (c == 255) c = 0;
				else c = 255;
			}

			imageR32Mask(x, y) = c;
		}
	}

	ColorImageR32 imageR32MaskLarge(image.getWidth(), image.getHeight());
	imageR32MaskLarge.setPixels(0);
	for (unsigned int y = 0; y < imageMask.getHeight(); y++) {
		for (unsigned int x = 0; x < imageMask.getWidth(); x++) {
			imageR32MaskLarge(x + offsetY, y + offsetX) = imageR32Mask(x, y);
		}
	}

    params.useCUDA = true;
    params.useThallo = true;
    params.nonLinearIter = 1;
    params.linearIter = 100;

    // This example has a couple solvers that don't fit into the CombinedSolverParameters mold.
    bool useCUDAPatch = false;
    bool useEigen = true;

    CombinedSolver solver(imageR32, image1Large, imageR32MaskLarge, params, useCUDAPatch, useEigen);
    solver.solveAll();
    ColorImageR32G32B32A32* res = solver.result();
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = math::round(math::clamp((*res)(x, y).x, 0.0f, 255.0f));
			unsigned char g = math::round(math::clamp((*res)(x, y).y, 0.0f, 255.0f));
			unsigned char b = math::round(math::clamp((*res)(x, y).z, 0.0f, 255.0f));
			out(x, y) = vec4uc(r, g, b,255);
		}
	}
	LodePNG::save(out, "output.png");
	return 0;
}