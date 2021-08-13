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
#include <Eigen/StdVector>

typedef Eigen::Matrix<float, 3, 1> Vector3f;
typedef Eigen::Matrix<float, 4, 1> Vector4f;

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector3f)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Vector4f)

void LoadVector(const std::string &filename, float *res, unsigned int length)
{
	std::ifstream in(filename, std::ifstream::in | std::ifstream::binary);
	if (!in)
	{
		std::cout << "ERROR:\tCan not open file: " << filename << std::endl;
		while (1);
	}
	unsigned int numberOfEntries;
	in.read((char*)&numberOfEntries, sizeof(unsigned int));
	if (length == 0) length = numberOfEntries;
	in.read((char*)(res), length*sizeof(float));

	in.close();
}
#include <tclap/CmdLine.h>
int main(int argc, char *argv[]) {
    bool performanceRun = false;
    CombinedSolverParameters params;
    params.autoschedulerSetting = 0;
    params.thallofile = "face_fitting.t";

    params.invasiveTiming = false;
    // Wrap everything in a try block.  Do this every time, 
    // because exceptions will be thrown for problems.
    try {
        TCLAP::CmdLine cmd("Thallo Example", ' ', "1.0");
        TCLAP::SwitchArg perfSwitch("p", "perf", "Performance Run", cmd);
        TCLAP::SwitchArg invasiveSwitch("i", "invasiveTiming", "Invasive Timing", cmd);
        TCLAP::ValueArg<int> autoArg("a", "autoschedule", "Autoschedule level", false, params.autoschedulerSetting, "int", cmd);
        TCLAP::ValueArg<std::string> thalloArg("o", "thallofile", "File to use for Thallo", false, params.thallofile, "string", cmd);

        // Parse the argv array.
        cmd.parse(argc, argv);


        // Get the value parsed by each arg.
        performanceRun = perfSwitch.getValue();
        params.autoschedulerSetting = autoArg.getValue();
        params.thallofile = thalloArg.getValue();
        params.invasiveTiming = invasiveSwitch.getValue();
    }
    catch (TCLAP::ArgException &e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

    const std::string data_dir = "../data/";

    std::string filename = data_dir + "average.off";
    std::string filenameTarget = data_dir + "target.off";

    std::string filenameBlendshapes(data_dir + "ExpressionBasisPCA.matrix");
    std::string filenameBlendshapeWeights(data_dir + "StandardDeviationExpressionPCA.vec");

    std::string filenameShape(data_dir + "ShapeBasisPCA.matrix");
    std::string filenameShapeWeights(data_dir + "StandardDeviationShapePCA.vec");

	unsigned int numberOfBlendshapes = 70;
    unsigned int numberOfShapes = numberOfBlendshapes;

	// Load Average Mesh
	SimpleMesh* mesh = new SimpleMesh();
	if (!OpenMesh::IO::read_mesh(*mesh, filename)) 
	{
	    std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << filename << std::endl;
		exit(1);
	}

	for (unsigned int i = 0; i < mesh->n_vertices(); ++i)
	{
		Vec3f p = mesh->point(VertexHandle(i));
		mesh->set_point(VertexHandle(i), p / 1000000.0f);
	}

	// Store Input
    if (!OpenMesh::IO::write_mesh(*mesh, data_dir + "input.off"))
	{
		std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	}

	Eigen::VectorXf a(3 * mesh->n_vertices());
	for (unsigned int i = 0; i < mesh->n_vertices(); ++i)
	{
		Vec3f p = mesh->point(VertexHandle(i));
		a(3 * i + 0) = p[0];
		a(3 * i + 1) = p[1];
		a(3 * i + 2) = p[2];
	}

	// Load Blendshapes
	Vector4f* inputBasis = new Eigen::Vector4f[mesh->n_vertices() * numberOfBlendshapes];
	Eigen::VectorXf inputStd(numberOfBlendshapes);

	LoadVector(filenameBlendshapeWeights, (float*)&inputStd(0), (unsigned int)numberOfBlendshapes);
	LoadVector(filenameBlendshapes, (float*)inputBasis, 4 * (unsigned int)mesh->n_vertices() * numberOfBlendshapes);
	Eigen::MatrixXf B(3 * mesh->n_vertices(), numberOfBlendshapes + numberOfShapes);
	for (unsigned int m = 0; m < numberOfBlendshapes; ++m)
	{
		for (unsigned int i = 0; i < mesh->n_vertices(); ++i)
		{
			B(3 * i + 0, m) = inputStd(m)*inputBasis[m*mesh->n_vertices() + i].x()*100.0f;
			B(3 * i + 1, m) = inputStd(m)*inputBasis[m*mesh->n_vertices() + i].y()*100.0f;
			B(3 * i + 2, m) = inputStd(m)*inputBasis[m*mesh->n_vertices() + i].z()*100.0f;
		}
	}

	// Load Identity
	Vector4f* inputBasisShape = new Eigen::Vector4f[mesh->n_vertices() * numberOfShapes];
	Eigen::VectorXf inputShapeStd(numberOfShapes);
	LoadVector(filenameShapeWeights, (float*)&inputShapeStd(0), (unsigned int)numberOfShapes);
	LoadVector(filenameShape, (float*)inputBasisShape, 4 * (unsigned int)mesh->n_vertices() * numberOfShapes);
	
	for (unsigned int m = 0; m < numberOfShapes; ++m)
	{
		for (unsigned int i = 0; i < mesh->n_vertices(); ++i)
		{
			B(3 * i + 0, m + numberOfBlendshapes) = inputShapeStd(m)*inputBasisShape[m*mesh->n_vertices() + i].x();
			B(3 * i + 1, m + numberOfBlendshapes) = inputShapeStd(m)*inputBasisShape[m*mesh->n_vertices() + i].y();
			B(3 * i + 2, m + numberOfBlendshapes) = inputShapeStd(m)*inputBasisShape[m*mesh->n_vertices() + i].z();
		}
	}
		
	// Compute Target

	//SimpleMesh* target = new SimpleMesh();
	//if (!OpenMesh::IO::read_mesh(*target, filenameTarget))
	//{
	//	std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
	//	std::cout << filename << std::endl;
	//	exit(1);
	//}

	Eigen::VectorXf d(numberOfBlendshapes + numberOfShapes); d.setZero();
	d(0) = 0.1f;
	d(numberOfBlendshapes + 0) = 2.0f;
	Eigen::VectorXf t = a + B*d;
	
	SimpleMesh target = *mesh;
	for (unsigned int i = 0; i < mesh->n_vertices(); ++i)
	{
		Vec3f p(t(3 * i + 0), t(3 * i + 1), t(3 * i + 2));
		target.set_point(VertexHandle(i), p);
	}
	target.request_face_normals();
	target.request_vertex_normals();
	target.update_face_normals();
	target.update_vertex_normals();
	
	unsigned int numberOfEdges = (uint)target.n_edges();
	float averageEdgeLength = 0.0f;
	for (auto edgeHandle : target.edges()) {
		auto edge = target.edge(edgeHandle);
		averageEdgeLength += target.calc_edge_length(edgeHandle) / numberOfEdges;
	}
	
	std::mt19937 rnd = std::mt19937(230948);
	float noiseModifier = 0.5f;
	std::normal_distribution<float> normalDistribution(0.0f, averageEdgeLength * noiseModifier);
	
	for (unsigned int i = 0; i < mesh->n_vertices(); ++i)
	{
		Vec3f p = target.point(VertexHandle(i));
		p += normalDistribution(rnd)*target.normal(VertexHandle(i));
		/* Re-enable for jitter in the data */ //target.set_point(VertexHandle(i), p);
	}
    printf("Blendshape Count: %u\nBasisshape Count: %u\n", numberOfBlendshapes, numberOfShapes);
	
	// Store Target
	if (!OpenMesh::IO::write_mesh(target, "target.off"))
	{
		std::cerr << "Error -> File: " << __FILE__ << " Line: " << __LINE__ << " Function: " << __FUNCTION__ << std::endl;
		std::cout << "out.off" << std::endl;
		exit(1);
	 }

    // f = 5903.7, k1 = 0.109306, k2 = -1.43375

    params.numIter = (performanceRun ? 2 : 30);
    params.nonLinearIter = 5;
    params.linearIter = 25;
	params.useThallo = true;
    params.useCUDA = performanceRun;
	params.thalloDoublePrecision = false;

    params.numIter = 1;

    //params.useThallo = false;
    //params.useThalloLM = true;

    CombinedSolver solver(mesh, &target, &B, params);
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
