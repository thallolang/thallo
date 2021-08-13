#include "main.h"
#include "CombinedSolver.h"

static void loadConstraints(std::vector<std::vector<int> >& constraints, std::string filename) {
  std::ifstream in(filename, std::fstream::in);

	if(!in.good())
	{
		std::cout << "Could not open marker file " << filename << std::endl;
		assert(false);
	}

	unsigned int nMarkers;
	in >> nMarkers;
	constraints.resize(nMarkers);
	for(unsigned int m = 0; m<nMarkers; m++)
	{
		int temp;
		for (int i = 0; i < 4; ++i) {
			in >> temp;
			constraints[m].push_back(temp);
		}

	}

	in.close();
}


#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {
    std::string filename = "../data/cat512.png";
    int downsampleFactor = 1;

    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 1;
    params.thallofile = "image_warping.t";
    params.invasiveTiming = false;
    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {
        TCLAP::CmdLine cmd("Thallo Example", ' ', "1.0");
        TCLAP::SwitchArg perfSwitch("p", "perf", "Performance Run", cmd);
        TCLAP::SwitchArg invasiveSwitch("i", "invasiveTiming", "Invasive Timing", cmd);
        TCLAP::ValueArg<int> autoArg("a", "autoschedule", "Autoschedule level", false, params.autoschedulerSetting, "int", cmd);
        TCLAP::ValueArg<std::string> thalloArg("o", "thallofile", "File to use for Thallo", false, params.thallofile, "string", cmd);
        TCLAP::UnlabeledValueArg<std::string> fileArg("file", "Filename", false, filename, "string", cmd);
        TCLAP::ValueArg<int> downsampleFactorArg("d", "downsampleFactor", "downsampleFactor", false, downsampleFactor, "int", cmd);

        // Parse the argv array.
        cmd.parse(argc, argv);


        // Get the value parsed by each arg.
        performanceRun              = perfSwitch.getValue();
        params.autoschedulerSetting = autoArg.getValue();
        params.thallofile              = thalloArg.getValue();
        params.invasiveTiming       = invasiveSwitch.getValue();
        filename                    = fileArg.getValue();
        downsampleFactor = downsampleFactorArg.getValue();
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }



	bool lmOnlyFullSolve = false;
    if (performanceRun && downsampleFactor > 0)
        lmOnlyFullSolve = true;
    downsampleFactor = std::max(1, downsampleFactor);

    // Must have a mask and constraints file in the same directory as the input image
    std::string maskFilename = filename.substr(0, filename.size() - 4) + "_mask.png";
    std::string constraintsFilename = filename.substr(0, filename.size() - 3) + "constraints";
    std::vector<std::vector<int>> constraints;
    loadConstraints(constraints, constraintsFilename);

    ColorImageR8G8B8A8 image = LodePNG::load(filename);
    const ColorImageR8G8B8A8 imageMask = LodePNG::load(maskFilename);

    ColorImageR32G32B32 imageColor(image.getWidth() / downsampleFactor, image.getHeight() / downsampleFactor);
    for (unsigned int y = 0; y < image.getHeight() / downsampleFactor; y++) {
        for (unsigned int x = 0; x < image.getWidth() / downsampleFactor; x++) {
            auto val = image(x*downsampleFactor, y*downsampleFactor);

            imageColor(x,y) = vec3f(val.x, val.y, val.z);
        }
    }

    ColorImageR32 imageR32(imageColor.getWidth(), imageColor.getHeight());
    printf("width %d, height %d\n", imageColor.getWidth(), imageColor.getHeight());
    for (unsigned int y = 0; y < imageColor.getHeight(); y++) {
        for (unsigned int x = 0; x < imageColor.getWidth(); x++) {
            imageR32(x, y) = imageColor(x, y).x;
		}
	}
    int activePixels = 0;

    ColorImageR32 imageR32Mask(imageMask.getWidth() / downsampleFactor, imageMask.getHeight() / downsampleFactor);
    for (unsigned int y = 0; y < imageMask.getHeight() / downsampleFactor; y++) {
        for (unsigned int x = 0; x < imageMask.getWidth() / downsampleFactor; x++) {
            imageR32Mask(x, y) = imageMask(x*downsampleFactor, y*downsampleFactor).x;
            if (imageMask(x*downsampleFactor, y*downsampleFactor).x == 0.0f) {
                ++activePixels;
            }
		}
	}
    printf("numActivePixels: %d\n", activePixels);
	
    for (auto& constraint : constraints) {
        for (auto& c : constraint) {
            c /= downsampleFactor;
        }
    }

    for (unsigned int y = 0; y < imageColor.getHeight(); y++)
	{
        for (unsigned int x = 0; x < imageColor.getWidth(); x++)
		{
            if (y == 0 || x == 0 || y == (imageColor.getHeight() - 1) || x == (imageColor.getWidth() - 1))
			{
				std::vector<int> v; v.push_back(x); v.push_back(y); v.push_back(x); v.push_back(y);
				constraints.push_back(v);
			}
		}
	}

    params.numIter = 19;
    params.useCUDA = false;
    params.nonLinearIter = 8;
    params.linearIter = 100;
    if (performanceRun) {
        params.useCUDA = true;
        params.useThallo = true;
        params.useThalloLM = false;
        params.earlyOut = true;
    }
    if (lmOnlyFullSolve) {
        params.useCUDA = false;
        params.useThallo = false;
        params.useThalloLM = true;
        params.linearIter = 500;
        if (image.getWidth() > 1024) {
            params.nonLinearIter = 100;
        }
    }
    params.useThallo = true;


	CombinedSolver solver(imageR32, imageColor, imageR32Mask, constraints, params);
    solver.solveAll();
    ColorImageR32G32B32* res = solver.result();
	ColorImageR8G8B8A8 out(res->getWidth(), res->getHeight());
	for (unsigned int y = 0; y < res->getHeight(); y++) {
		for (unsigned int x = 0; x < res->getWidth(); x++) {
			unsigned char r = util::boundToByte((*res)(x, y).x);
            unsigned char g = util::boundToByte((*res)(x, y).y);
            unsigned char b = util::boundToByte((*res)(x, y).z);
			out(x, y) = vec4uc(r, g, b, 255);
	
			for (unsigned int k = 0; k < constraints.size(); k++)
			{
				if (constraints[k][2] == x && constraints[k][3] == y) 
				{
                    if (imageR32Mask(constraints[k][0], constraints[k][1]) == 0)
					{
						//out(x, y) = vec4uc(255, 0, 0, 255);
					}
				}
		
				if (constraints[k][0] == x && constraints[k][1] == y)
				{
					if (imageR32Mask(x, y) == 0)
					{
                        image(x*downsampleFactor, y*downsampleFactor) = vec4uc(255, 0, 0, 255);
					}
				}
			}
		}
	}
	LodePNG::save(out, "output.png");
	LodePNG::save(image, "inputMark.png");
    printf("Saved\n");

	return 0;
}
