#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"
#include "LandMarkSet.h"
#include <string>
#include <OpenMesh/Tools/Subdivider/Uniform/SubdividerT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LongestEdgeT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/LoopT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/CatmullClarkT.hh>
#include <OpenMesh/Tools/Subdivider/Uniform/Sqrt3T.hh>
#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {

	std::string filename = "../data/small_armadillo.ply";
    int subdivisionFactor = -1;

    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 1;
    params.thallofile = "arap_mesh_deformation.t";
    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {
        TCLAP::CmdLine cmd("Thallo Example", ' ', "1.0");
        TCLAP::SwitchArg perfSwitch("p", "perf", "Performance Run", cmd);
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
        filename = fileArg.getValue();
        subdivisionFactor = subArg.getValue();
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    // For now, any model must be accompanied with a identically 
    // named (besides the extension, which must be 3 characters) mrk file
    std::string markerFilename = filename.substr(0, filename.size() - 3) + "mrk";

    bool lmOnlyFullSolve = (subdivisionFactor >= 0);

	// Load Constraints
	LandMarkSet markersMesh;
    markersMesh.loadFromFile(markerFilename.c_str());

	std::vector<int>				constraintsIdx;
	std::vector<std::vector<float>> constraintsTarget;

	for (unsigned int i = 0; i < markersMesh.size(); i++)
	{
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

    params.numIter = 10;
    params.nonLinearIter = 20;
    params.linearIter = 100;
    params.useThallo = true;
    if (performanceRun) {
        params.useCUDA  = true;
        params.useThallo   = true;
        params.useThalloLM = false;
        params.earlyOut = true;
        params.nonLinearIter = 20;
        params.linearIter = 1000;
    }
    if (lmOnlyFullSolve) {
        params.useCUDA = false;
        params.useThallo = false;
        params.useThalloLM = true;
        params.earlyOut = true;
        params.linearIter = 1000;// m_image.getWidth()*m_image.getHeight();
        if (mesh->n_vertices() > 100000) {
            params.nonLinearIter = (unsigned int)mesh->n_vertices() / 5000;
        }
    }
    params.thalloDoublePrecision = false;

    float weightFit = 4.0f;
    float weightReg = 1.0f;
    CombinedSolver solver(mesh, constraintsIdx, constraintsTarget, params, weightFit, weightReg);
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
