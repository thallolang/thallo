#pragma once
#include <cmath>
#include "ThalloSolver.h"
#include "CombinedSolverParameters.h"
#include "SolverIteration.h"
#include "Config.h"

static std::string qt(std::string str) { return "\"" + str + "\""; }

static void toStream(std::string name, SolverPerformanceEntry& entry, std::ostream& out, std::string ident, bool commaAfter = true) {
    std::string ender = commaAfter ? "," : "";
    out << ident + qt(name) + " : {" << std::endl;
    std::string newIdent = ident + "  ";
    auto emit = [&](std::string fieldname, double field, std::string ending) {
      field = std::isnan(field) ? 9999999999999999999999.0 : field;
      out << newIdent << qt(fieldname) + " : " << field << ending << std::endl; };
    out << newIdent << qt("count") + " : " << entry.count << "," << std::endl;
    emit("minMS", entry.minMS, ",");
    emit("maxMS", entry.maxMS, ",");
    emit("meanMS", entry.meanMS,",");
    emit("stddevMS", entry.stddevMS,"");
    out << ident << "}" << ender << std::endl;
}

static void toStream(SolverPerformanceSummary& summary, std::ostream& out, std::string ident) {
    std::string newIdent = ident + "  ";
    out << "{" << std::endl;
    toStream("total",               summary.total,              out, newIdent);
    toStream("nonlinearIteration",  summary.nonlinearIteration, out, newIdent);
    toStream("nonlinearSetup",      summary.nonlinearSetup,     out, newIdent);
    toStream("linearSolve",         summary.linearSolve,        out, newIdent);
    toStream("nonlinearResolve",    summary.nonlinearResolve,   out, newIdent, false);
    out << ident << "}";
}

/** We want to run several solvers in an identical manner, with some initalization
and finish code for each of the examples. The structure is the same for every
example, so we keep it in solveAll(), and let individual examples override
combinedSolveInit(); combinedSolveFinalize(); preSingleSolve(); postSingleSolve();*/
class CombinedSolverBase {
public:
    virtual void combinedSolveInit() = 0;
    virtual void combinedSolveFinalize() = 0;
    virtual void preSingleSolve() = 0;
    virtual void postSingleSolve() = 0;
    virtual void preNonlinearSolve(int iteration) = 0;
    virtual void postNonlinearSolve(int iteration) = 0;

    CombinedSolverBase(std::string name) : m_name(name) {}

    void reportFinalSolverCosts(std::string name, std::ostream& output = std::cout) {
        reportFinalCosts(name, m_combinedSolverParameters, getCost("ThalloGN"), getCost("ThalloLM"), getCost("Ceres"), output);
    }

    void saveFinalCosts(std::string name) {
        std::ofstream ofs("finalCosts.json");
        if (ofs.good()) {
            reportFinalSolverCosts(name, ofs);
        } else {
            std::cout << "Error opening finalCosts.json" << std::endl;
        }
    }

    void reportPerformanceStatistics(std::string name, std::ostream& output = std::cout) {
        output << "{  \"name\" : \"" << name << "\"," << std::endl;
        auto a = m_combinedSolverParameters.autoschedulerSetting;
        std::string autoscheduled = (a == 2) ? "2" : ((a == 1) ? "1" : "0");
        output << "  \"autoscheduled\" : " << autoscheduled << "," << std::endl;
        output << "  \"performance\" : {" << std::endl;

        output << std::scientific;
        output << std::setprecision(18);

        std::vector<std::pair<std::string, SolverPerformanceSummary>> perf;

        if (m_combinedSolverParameters.useThallo)      perf.push_back({ "ThalloGN", getPerfStats("ThalloGN") });
        if (m_combinedSolverParameters.useThalloLM)    perf.push_back({ "ThalloLM", getPerfStats("ThalloLM") });
        if (m_combinedSolverParameters.useCUDA)     perf.push_back({ "Cuda", getPerfStats("Cuda") });
        auto maybeEigen = getPerfStats("Eigen");
        if (maybeEigen.total.count > 0) {
            perf.push_back({ "Ceres", maybeEigen });
        }

        for (int i = 0; i < perf.size(); ++i) {
            auto delim = (i != perf.size() - 1) ? "," : "";
            output << "    \"" << perf[i].first << "\" : ";
            toStream(perf[i].second, output, "    ");
            output << delim << std::endl;
        }
        output << "  }" << std::endl << "}" << std::endl;
    }

    void savePerformanceStatistics(std::string name) {
        std::ofstream ofs("perf.json");
        if (ofs.good()) {
            reportPerformanceStatistics(name, ofs);
        } else {
            std::cout << "Error opening perf.json" << std::endl;
        }
    }

    virtual void solveAll() {
        combinedSolveInit();
        for (auto& s : m_solverInfo) {
            if (s.enabled) {
                singleSolve(s, m_solverParams, m_problemParams);
            }
        }
        combinedSolveFinalize();
        if (m_combinedSolverParameters.profileSolve) {
            ceresIterationComparison(m_name, m_combinedSolverParameters.thalloDoublePrecision);
        }
        saveFinalCosts(m_name);
        savePerformanceStatistics(m_name);
    }

    double getCost(std::string name) {
        for (auto s : m_solverInfo) {
            if (s.name == name) {
                if (s.solver && s.enabled) {
                    return s.solver->finalCost();
                }
            }
        }
        return nan("");
    }
    SolverPerformanceSummary getPerfStats(std::string name) {
        for (auto s : m_solverInfo) {
            if (s.name == name) {
                if (s.solver && s.enabled) {
                    return s.solver->getSummaryStatistics();
                }
            }
        }
        return{};
    }

    void setParameters(const CombinedSolverParameters& params) {
        m_combinedSolverParameters = params;
    }

    std::vector<SolverIteration> getIterationInfo(std::string name) {
        for (auto& s : m_solverInfo) {
            if (s.name == name) {
                if (s.solver && s.enabled) {
                    return s.iterationInfo;
                }
            }
        }
        return std::vector<SolverIteration>();
    }

    void ceresIterationComparison(std::string name, bool thalloDoublePrecision) {
        saveSolverResults("results/", thalloDoublePrecision ? "_double" : "_float", getIterationInfo("Ceres"), getIterationInfo("ThalloGN"), getIterationInfo("ThalloLM"), thalloDoublePrecision);
    }

    void addSolver(std::shared_ptr<SolverBase> solver, std::string name, bool enabled = true) {
        m_solverInfo.resize(m_solverInfo.size() + 1);
        m_solverInfo[m_solverInfo.size() - 1].set(solver, name, enabled);

    }

    void addThalloSolvers(std::vector<unsigned int> dims, const CombinedSolverParameters& p) {
        if (p.useThallo) {
            addSolver(std::make_shared<ThalloSolver>(dims, p.thallofile, "gauss_newton", p.thalloDoublePrecision, p.invasiveTiming, p.autoschedulerSetting, p.thalloCPU), "ThalloGN", true);
        }
        if (p.useThalloLM) {
            if (p.autoschedulerSetting <= 2 || !p.useThallo) // TODO: remove; this is for exhaustive autoscheduling experiment
                addSolver(std::make_shared<ThalloSolver>(dims, p.thallofile, "levenberg_marquardt", p.thalloDoublePrecision, p.invasiveTiming, p.autoschedulerSetting, p.thalloCPU), "ThalloLM", true);
        }
    }

    void addThalloSolvers(const std::vector<unsigned int>& dims) {
        addThalloSolvers(dims, m_combinedSolverParameters);
    }

    std::string activeSolverName() const {
        return m_activeSolverInfo.name;
    }

protected:
    struct SolverInfo {
        std::shared_ptr<SolverBase> solver;
        std::vector<SolverIteration> iterationInfo;
        std::string name;
        bool enabled;
        SolverPerformanceSummary perfSummary;
        void set(std::shared_ptr<SolverBase> _solver, std::string _name, bool _enabled) {
            solver = std::move(_solver);
            name = _name;
            enabled = _enabled;
        }
    };
    std::vector<SolverInfo> m_solverInfo;

    virtual void singleSolve(SolverInfo& s, const NamedParameters& solverParams, const NamedParameters& problemParams) {
        m_activeSolverInfo = s;
        preSingleSolve();
        if (m_combinedSolverParameters.numIter == 1) {
            preNonlinearSolve(0);
            std::cout << "//////////// (" << s.name << ") ///////////////" << std::endl;
            s.solver->solve(solverParams, problemParams, s.perfSummary, m_combinedSolverParameters.profileSolve, s.iterationInfo);
            postNonlinearSolve(0);
        } else {
            for (int i = 0; i < (int)m_combinedSolverParameters.numIter; ++i) {
                std::cout << "//////////// ITERATION" << i << "  (" << s.name << ") ///////////////" << std::endl;
                preNonlinearSolve(i);
                s.solver->solve(solverParams, problemParams, s.perfSummary, m_combinedSolverParameters.profileSolve, s.iterationInfo);
                postNonlinearSolve(i);
                if (m_combinedSolverParameters.earlyOut || m_endSolveEarly) {
                    m_endSolveEarly = false;
                    break;
                }
            }
        }
        postSingleSolve();
    }
    SolverInfo m_activeSolverInfo;
    // Set to true in preNonlinearSolve or postNonlinearSolve to finish the solve before the specified number of iterations
    bool m_endSolveEarly = false;
    NamedParameters m_solverParams;
    NamedParameters m_problemParams;
    CombinedSolverParameters m_combinedSolverParameters;
    std::string m_name = "Default";
};
