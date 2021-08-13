#include "mLibInclude.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"
#include "LandMarkSet.h"

#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {
    std::string filename = "../data/raptor_simplify2k.off";

    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 1;
    params.thallofile = "embedded_mesh_deformation.t";

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

    // For now, any model must be accompanied with a identically 
    // named (besides the extension, which must be 3 characters) mrk file
    std::string markerFilename = filename.substr(0, filename.size() - 3) + "mrk";
	// Load Constraints
	LandMarkSet markersMesh;
    markersMesh.loadFromFile(markerFilename.c_str());

	std::vector<int>				constraintsIdx;
	std::vector<std::vector<float>> constraintsTarget;

	for (unsigned int i = 0; i < markersMesh.size(); i++)
	{
        printf("%d: %d\n", i, (markersMesh[i].getVertexIndex()));
		constraintsIdx.push_back(markersMesh[i].getVertexIndex());
		constraintsTarget.push_back(markersMesh[i].getPosition());
	}

	SimpleMesh* mesh = new SimpleMesh();
	if (!OpenMesh::IO::read_mesh(*mesh, filename))
	{
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename << std::endl;
		exit(1);
	}
	printf("Beginning MeshDeformationED Warp\n");

    /*params.useThallo = true;
    params.useThalloLM = false;
    params.numIter = 32;
    params.nonLinearIter = 1;
    params.linearIter = 4000;
    params.earlyOut = false;
    */

    /* LM is good here */
    params.useThallo = true;
    params.useThalloLM = false;
    params.numIter = 31;
    params.nonLinearIter = 5;
    params.linearIter = 125;

    CombinedSolver solver(mesh, constraintsIdx, constraintsTarget, params);
    solver.solveAll();
    SimpleMesh* res = solver.result();

	if (!OpenMesh::IO::write_mesh(*res, "out.off"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}
	return 0;
}
