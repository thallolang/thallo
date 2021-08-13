#include "mLibInclude.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {
    std::string filename = "../data/head.ply";

    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 1;
    params.thallofile = "volumetric_mesh_deformation.t";
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


	SimpleMesh* mesh = new SimpleMesh();

	if (!OpenMesh::IO::read_mesh(*mesh, filename)) 
	{
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << filename << std::endl;
		exit(1);
	}
	printf("Faces: %d\nVertices: %d\n", (int)mesh->n_faces(), (int)mesh->n_vertices());

    params.nonLinearIter = 20;
    params.linearIter = 60;

    params.useThallo = true;
    params.useCUDA = performanceRun;

    int3 voxelGridSize = make_int3(5, 20, 5);
    CombinedSolver solver(mesh, voxelGridSize, params);
    solver.solveAll();
    SimpleMesh* res = solver.result();
    solver.saveGraphResults();

	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}

	return 0;
}
