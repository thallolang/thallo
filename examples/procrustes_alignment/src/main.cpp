#include "main.h"
#include "CombinedSolver.h"
#include "../../shared/OpenMesh.h"
#include <string>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LongestEdgeT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LoopT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/CatmullClarkT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/Sqrt3T.hh>
#include <Eigen/Core>
#include <Eigen/Geometry> 

#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {
    std::string filename = "../data/raptor_simplify2k.off";
    int subdivisionFactor = 0;

    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 1;
    params.thallofile = "procrustes_alignment.t";

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
        TCLAP::ValueArg<int> subArg("s", "subdivision", "Subdivision Factor", false, subdivisionFactor, "int", cmd);

        // Parse the argv array.
        cmd.parse(argc, argv);


        // Get the value parsed by each arg.
        performanceRun = perfSwitch.getValue();
        params.autoschedulerSetting = autoArg.getValue();
        params.thallofile = thalloArg.getValue();
        params.invasiveTiming = invasiveSwitch.getValue();
        filename = fileArg.getValue();
        subdivisionFactor = subArg.getValue();
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }


	bool lmOnlyFullSolve = (subdivisionFactor > 0);

	// Load Mesh
	SimpleMesh* mesh = new SimpleMesh();
	if (!OpenMesh::IO::read_mesh(*mesh, filename)) 
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << filename << std::endl;
		exit(1);
	}

	OpenMesh::Subdivider::Uniform::Sqrt3T<SimpleMesh> subdivider;
	// Initialize subdivider
	if (lmOnlyFullSolve) {
		if (subdivisionFactor > 0) {
			subdivider.attach(*mesh);
			subdivider(subdivisionFactor);
			subdivider.detach();
		}
	} else {
		//OpenMesh::Subdivider::Uniform::CatmullClarkT<SimpleMesh> catmull;
		// Execute 1 subdivision steps
		subdivider.attach(*mesh);
		subdivider(1);
		subdivider.detach();
	}
	printf("Faces: %d\nVertices: %d\n", (int)mesh->n_faces(), (int)mesh->n_vertices());

	// Compute Target
	Eigen::Matrix3f T = Eigen::AngleAxisf(0.25*M_PI, Eigen::Vector3f::UnitX()).toRotationMatrix();

	SimpleMesh target = *mesh;
	for (unsigned int i = 0; i < target.n_vertices(); i++) {
		SimpleMesh::Point p = target.point(VertexHandle(i));
		Eigen::Vector3f pE(p[0], p[1], p[2]);
		pE = T*pE;
		target.set_point(VertexHandle(i), SimpleMesh::Point(pE[0], pE[1], pE[2]));
	}

	// Store Target
	if (!OpenMesh::IO::write_mesh(target, "target.ply"))
	{
		std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}
    params.nonLinearIter = 10;
    params.linearIter = 6;
	params.useThallo = true;
    if (performanceRun) {
        //params.useCUDA = true;
        params.useThallo = true;
        //params.useThalloLM = true;
    }
    if (lmOnlyFullSolve) {
        params.useCUDA = false;
        params.useThallo = false;
        params.useThalloLM = true;
    }

    params.thalloDoublePrecision = false;

    CombinedSolver solver(mesh, &target, params);
    solver.solveAll();
    SimpleMesh* res = solver.result();
	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	    std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}

	return 0;
}
