#pragma once
#include <limits>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <string>
#include <ostream>
#include <cmath>
#include "CombinedSolverParameters.h"
using std::isnan;
struct SolverIteration
{
    SolverIteration() {}
    SolverIteration(double _cost, double _timeInMS) { cost = _cost; timeInMS = _timeInMS; }
    double cost = -std::numeric_limits<double>::infinity();
    double timeInMS = -std::numeric_limits<double>::infinity();
};

template<class T>
const T& clampedRead(const std::vector<T> &v, int index)
{
    if (index < 0) return v[0];
    if (index >= v.size()) return v[v.size() - 1];
    return v[index];
}

static void saveSolverResults(std::string directory, std::string suffix,
    const std::vector<SolverIteration>& ceresIters, const std::vector<SolverIteration>& thalloGNIters, const std::vector<SolverIteration>& thalloLMIters, bool thalloDoublePrecision) {
    std::ofstream resultFile(directory + "results" + suffix + ".csv");
    resultFile << std::scientific;
    resultFile << std::setprecision(20);
    std::string colSuffix = thalloDoublePrecision ? " (double)" : " (float)";
	resultFile << "Iter, Ceres Error, ";
	resultFile << "Thallo(GN) Error" << colSuffix << ",  Thallo(LM) Error" << colSuffix << ", Ceres Iter Time(ms), ";
	resultFile << "Thallo(GN) Iter Time(ms)" << colSuffix << ", Thallo(LM) Iter Time(ms)" << colSuffix << ", Total Ceres Time(ms), ";
	resultFile << "Total Thallo(GN) Time(ms)" << colSuffix << ", Total Thallo(LM) Time(ms)" << colSuffix << std::endl;
    double sumThalloGNTime = 0.0;
    double sumThalloLMTime = 0.0;
    double sumCeresTime = 0.0;

    auto _ceresIters = ceresIters;
    auto _thalloLMIters = thalloLMIters;
    auto _thalloGNIters = thalloGNIters;
    
    if (_ceresIters.size() == 0) {
        _ceresIters.push_back(SolverIteration(0, 0));
    }
    if (_thalloLMIters.size() == 0) {
        _thalloLMIters.push_back(SolverIteration(0, 0));
    }
    if (_thalloGNIters.size() == 0) {
        _thalloGNIters.push_back(SolverIteration(0, 0));
    }
    for (int i = 0; i < (int)std::max((int)_ceresIters.size(), std::max((int)_thalloLMIters.size(), (int)_thalloGNIters.size())); i++)
    {
        double ceresTime = ((_ceresIters.size() > i) ? _ceresIters[i].timeInMS : 0.0);
        double thalloGNTime = ((_thalloGNIters.size() > i) ? _thalloGNIters[i].timeInMS : 0.0);
        double thalloLMTime = ((_thalloLMIters.size() > i) ? _thalloLMIters[i].timeInMS : 0.0);
        sumCeresTime += ceresTime;
        sumThalloGNTime += thalloGNTime;
        sumThalloLMTime += thalloLMTime;
        resultFile << i << ", " << clampedRead(_ceresIters, i).cost << ", " << clampedRead(_thalloGNIters, i).cost << ", " << clampedRead(_thalloLMIters, i).cost << ", " << ceresTime << ", " << thalloGNTime << ", " << thalloLMTime << ", " << sumCeresTime << ", " << sumThalloGNTime << ", " << sumThalloLMTime << std::endl;
    }
}


static void reportFinalCosts(std::string name, const CombinedSolverParameters& params, double gnCost, double lmCost, double ceresCost, std::ostream& output = std::cout) {
    output << "{  \"name\" : \"" << name << "\"," << std::endl;
    output << "  \"costs\" : {" << std::endl;

    output << std::scientific;
    output << std::setprecision(20);

    std::vector<std::pair<std::string, double>> costs;

    if (params.useThallo && !isnan(gnCost)) costs.push_back({"ThalloGN", gnCost});
    if (params.useThalloLM && !isnan(lmCost)) costs.push_back({ "ThalloLM", lmCost });
    
    for (int i = 0; i < costs.size(); ++i) {
        auto delim = (i != costs.size()-1) ? "," : "";
        output << "    \"" << costs[i].first << "\" : " << costs[i].second << delim << std::endl;
    }
    output << "  }" << std::endl << "}" << std::endl;
}
