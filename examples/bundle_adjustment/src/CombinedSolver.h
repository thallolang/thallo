#pragma once

#include "mLibInclude.h"

#include <cuda_runtime.h>

#include "../../shared/CombinedSolverBase.h"
#include "../../shared/CombinedSolverParameters.h"
#include "../../shared/SolverIteration.h"
#include "bal_problem.h"

template <typename T>
static void updateThalloImage(std::shared_ptr<ThalloImage> dst, BaseImage<T> src) {
    dst->update((void*)src.getData(), sizeof(T)*src.getWidth()*src.getHeight(), ThalloImage::Location::CPU);
}

class CombinedSolver : public CombinedSolverBase {
public:
    CombinedSolver(std::shared_ptr<BALProblem> bal_problem, const CombinedSolverParameters& params, std::string name, bool runToConvergence=false, float timeLimit=3600.0f, float eta=0.1f) : CombinedSolverBase("Bundle Adjustment"),
        m_problemName(name), m_bal_problem(bal_problem), m_runToConvergence(runToConvergence), m_timeLimit(timeLimit), m_eta(eta){
        m_combinedSolverParameters = params;

        uint C = m_bal_problem->num_cameras();
        uint P = m_bal_problem->num_points();
        uint O = m_bal_problem->num_observations();
        m_dims = { C, P, O };
        
        std::vector<float> obs;
        m_bal_problem->GetFloatParameters(obs, m_cpuCameras, m_cpuPoints);

        auto createImFromFloats = [](std::vector<float>& vals, uint dim, uint channelCount, bool isUnknown){
            auto result = createEmptyThalloImage({ dim }, ThalloImage::Type::FLOAT, channelCount, ThalloImage::GPU, isUnknown);
            result->update(vals.data(), vals.size()*sizeof(float), ThalloImage::Location::CPU);
            return result;
        };
        
        m_observations = createImFromFloats(obs, O, 2, false); 
        m_cameraParams = createImFromFloats(m_cpuCameras, C, m_bal_problem->camera_block_size(), true);
        m_pointParams  = createImFromFloats(m_cpuPoints, P, 3, true);

        std::vector<int> cameraIndices, pointIndices;
        cameraIndices.resize(O);
        pointIndices.resize(O);
        const int* pi = m_bal_problem->point_index();
        const int* ci = m_bal_problem->camera_index();

        for (uint i = 0; i < O; ++i) {
            cameraIndices[i] = ci[i];
            pointIndices[i] = pi[i];
        }

        m_G = std::make_shared<ThalloGraph>(std::vector<std::vector<int> >({ cameraIndices, pointIndices }));
		reset();
        // Adds Thallo solvers according to settings in m_combinedSolverParameters
        if (m_runToConvergence) {
            int autosched = params.autoschedulerSetting;
            bool invasiveTiming = false;
            addSolver(std::make_shared<ThalloSolver>(m_dims, params.thallofile, "levenberg_marquardt", false, invasiveTiming, autosched), "ThalloLM_float");
            addSolver(std::make_shared<ThalloSolver>(m_dims, params.thallofile, "levenberg_marquardt", true, invasiveTiming, autosched), "ThalloLM_double");
            m_combinedSolverParameters.nonLinearIter = 1000000;
            m_combinedSolverParameters.linearIter = 1000000;
        } else {
            addThalloSolvers(m_dims);
        }

	}

    virtual void solveAll() override {
        combinedSolveInit();
        for (auto& s : m_solverInfo) {
            if (s.enabled) {
                singleSolve(s, m_solverParams, m_problemParams);
            }
        }
        combinedSolveFinalize();
        saveAllSolverResults("results/", m_problemName);
        saveFinalCosts(m_name);
        savePerformanceStatistics(m_name);
    }

    void saveAllSolverResults(std::string directory, std::string name) {
        std::vector<std::vector<SolverIteration>> iterations;
        std::vector<std::string> names;
        int maxIters = 0;
        for (auto& s : m_solverInfo) {
            if (s.solver && s.enabled) {
	      printf("%s initial time: %g\n", s.name.c_str(), s.iterationInfo[0].timeInMS);
	        iterations.push_back(s.iterationInfo);
                names.push_back(s.name);
                maxIters = std::max(maxIters,(int)s.iterationInfo.size());
            }
        }
        std::vector<double> sumTimes;
        sumTimes.resize(names.size());
        for (int s = 0; s < sumTimes.size(); ++s) {
            sumTimes[s] = 0.0;
        }

        std::ofstream resultFile(directory + name + ".csv");
        resultFile << std::scientific;
        resultFile << std::setprecision(20);
        resultFile << "Iter";

        for (int s = 0; s < names.size(); ++s) {
            resultFile << ", " << names[s] << " Error";
            resultFile << ", " << names[s] << " Time (ms)";
            resultFile << ", " << names[s] << " Total Time(ms)";
        }
        resultFile << std::endl;

        for (int it = 0; it < maxIters; it++) {
            resultFile << it;
            for (int s = 0; s < names.size(); ++s) {
                double time = ((iterations[s].size() > it) ? iterations[s][it].timeInMS : 0.0);
                sumTimes[s] += time;
                resultFile << ", " << clampedRead(iterations[s], it).cost;
                resultFile << ", " << time;
                resultFile << ", " << sumTimes[s];
            }
            resultFile << std::endl;
        }
    }

    virtual void combinedSolveInit() override {
        // Set in the same order as indices in param declaration
        m_problemParams.set("cameras", m_cameraParams);
        m_problemParams.set("points", m_pointParams);
        m_problemParams.set("observations", m_observations);
        m_problemParams.set("G", m_G);

        m_solverParams.set("nIterations", &m_combinedSolverParameters.nonLinearIter);
        m_solverParams.set("lIterations", &m_combinedSolverParameters.linearIter);

        static float function_tolerance = 0.0f;
        m_solverParams.set("q_tolerance", &m_eta);
        m_solverParams.set("function_tolerance", &function_tolerance);
        m_solverParams.set("max_solver_time_in_seconds", &m_timeLimit);
    }

    virtual void preSingleSolve() override {
        reset();
    }
    virtual void postSingleSolve() override {
        std::vector<float> camParams,ptParams;
        camParams.resize(m_cpuCameras.size());
        ptParams.resize(m_cpuPoints.size());
        m_cameraParams->copyTo(camParams);
        m_pointParams->copyTo(ptParams);
        m_bal_problem->SetFloatParameters(camParams, ptParams);
        m_bal_problem->WriteToPLYFile("after_"+m_activeSolverInfo.name+".ply");
    }
    virtual void preNonlinearSolve(int) override {}
    virtual void postNonlinearSolve(int) override {}

    virtual void combinedSolveFinalize() override {}

    void reset() {
        m_bal_problem->SetFloatParameters(m_cpuCameras, m_cpuPoints);
        m_cameraParams->update(m_cpuCameras.data(), m_cpuCameras.size()*sizeof(float), ThalloImage::Location::CPU);
        m_pointParams->update(m_cpuPoints.data(), m_cpuPoints.size()*sizeof(float), ThalloImage::Location::CPU);
	}

private:
    float m_timeLimit;
    float m_eta;
    bool m_runToConvergence;

    std::string m_problemName;

    std::vector<uint> m_dims;

    std::vector<float> m_cpuCameras;
    std::vector<float> m_cpuPoints;

    std::shared_ptr<ThalloImage> m_cameraParams;
    std::shared_ptr<ThalloImage> m_pointParams;
    std::shared_ptr<ThalloImage> m_observations;
    std::shared_ptr<ThalloGraph> m_G;

    std::shared_ptr<BALProblem> m_bal_problem;
	

};
