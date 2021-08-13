#include "main.h"
#include "CombinedSolver.h"
#include "OpenMesh.h"

static SimpleMesh* createMesh(std::string filename) {
    SimpleMesh* mesh = new SimpleMesh();
    if (!OpenMesh::IO::read_mesh(*mesh, filename))
    {
        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
        std::cout << filename << std::endl;
        exit(1);
    }
    printf("Faces: %d\nVertices: %d\n", (int)mesh->n_faces(), (int)mesh->n_vertices());
    return mesh;
}

static std::vector<int4> getSourceTetIndices(std::string filename) {
    // TODO: error handling
    std::ifstream inFile(filename);
    int tetCount = 0;
    int temp;
    inFile >> tetCount >> temp >> temp;
    std::vector<int4> tets(tetCount);
    for (int i = 0; i < tetCount; ++i) {
        inFile >> temp >> tets[i].x >> tets[i].y >> tets[i].z >> tets[i].w;
    }
    int4 f = tets[tets.size() - 1];
    printf("Final tet read: %d %d %d %d\n", f.x, f.y, f.z, f.w);
    return tets;
}

#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {
    std::string targetSourceDirectory = "../data/squat_target";
    std::string sourceFilename = "../data/squat_source.obj";
    std::string tetmeshFilename = "../data/squat_tetmesh.ele";
    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 1;
    params.thallofile = "robust_nonrigid_alignment.t";
    params.invasiveTiming = false;
    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {
        TCLAP::CmdLine cmd("Thallo Example", ' ', "1.0");
        TCLAP::SwitchArg perfSwitch("p", "perf", "Performance Run", cmd);
        TCLAP::SwitchArg invasiveSwitch("i", "invasiveTiming", "Invasive Timing", cmd);
        TCLAP::ValueArg<int> autoArg("a", "autoschedule", "Autoschedule level", false, params.autoschedulerSetting, "int", cmd);
        TCLAP::ValueArg<std::string> thalloArg("o", "thallofile", "File to use for Thallo", false, params.thallofile, "string", cmd);
        TCLAP::UnlabeledMultiArg<std::string> inputArg("inputs", "tar src tet", false, "multi string", cmd);

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
            targetSourceDirectory = inputs[0];
            sourceFilename = inputs[1];
            tetmeshFilename = inputs[2];
        }

    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    std::vector<std::string> targetFiles = ml::Directory::enumerateFiles(targetSourceDirectory);

    std::vector<int4> sourceTetIndices = getSourceTetIndices(tetmeshFilename);

    SimpleMesh* sourceMesh = createMesh(sourceFilename);

    std::vector<SimpleMesh*> targetMeshes;
    for (auto target : targetFiles) {
        targetMeshes.push_back(createMesh(targetSourceDirectory + "/" + target));
    }
    std::cout << "All meshes now in memory" << std::endl;

    params.numIter = 15;
    params.nonLinearIter = 10;
    params.linearIter = 250;
    params.useThallo = false;
    params.useThalloLM = true;
    if (params.autoschedulerSetting > 2) {
        params.useThallo = true;
        params.useThalloLM = false;
        params.numIter = 1;
    }


    CombinedSolver solver(sourceMesh, targetMeshes, sourceTetIndices, params);
    solver.solveAll();
    SimpleMesh* res = solver.result();
    
	if (!OpenMesh::IO::write_mesh(*res, "out.ply"))
	{
	        std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}
    
    for (SimpleMesh* mesh : targetMeshes) {
        delete mesh;
    }
    delete sourceMesh;

	return 0;
}
