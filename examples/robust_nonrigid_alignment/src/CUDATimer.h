#ifndef CUDATimer_h
#define CUDATimer_h
#include <cuda_runtime.h>
#include <string>
#include <vector>
struct TimingInfo {
    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    float duration;
    std::string eventName;
};

/** Copied wholesale from mLib, so nvcc doesn't choke. */
template<class T>
int findFirstIndex(const std::vector<T> &collection, const T &value)
{
    int index = 0;
    for (const auto &element : collection)
    {
        if (element == value)
            return index;
        index++;
    }
    return -1;
}

struct CUDATimer {
    std::vector<TimingInfo> timingEvents;
    int currentIteration;

    CUDATimer() : currentIteration(0) {
        TimingInfo timingInfo;
        cudaEventCreate(&timingInfo.startEvent);
        cudaEventCreate(&timingInfo.endEvent);
        cudaEventRecord(timingInfo.startEvent);
        timingInfo.eventName = "overall";
        timingEvents.push_back(timingInfo);
    }
    void nextIteration() {
        ++currentIteration;
    }

    void reset() {
        currentIteration = 0;
        timingEvents.clear();
    }

    void startEvent(const std::string& name) {
        TimingInfo timingInfo;
        cudaEventCreate(&timingInfo.startEvent);
        cudaEventCreate(&timingInfo.endEvent);
        cudaEventRecord(timingInfo.startEvent);
        timingInfo.eventName = name;
        timingEvents.push_back(timingInfo);
    }

    void endEvent() {
        TimingInfo& timingInfo = timingEvents[timingEvents.size() - 1];
        cudaEventRecord(timingInfo.endEvent, 0);
    }

    void evaluate() {
        cudaEventRecord(timingEvents[0].endEvent);
        std::vector<std::string> aggregateTimingNames;
        std::vector<float> aggregateTimes;
        std::vector<int> aggregateCounts;
        for (int i = 0; i < timingEvents.size(); ++i) {
            TimingInfo& eventInfo = timingEvents[i];
            cudaEventSynchronize(eventInfo.endEvent);
            cudaEventElapsedTime(&eventInfo.duration, eventInfo.startEvent, eventInfo.endEvent);
            int index = findFirstIndex(aggregateTimingNames, eventInfo.eventName);
            if (index < 0) {
                aggregateTimingNames.push_back(eventInfo.eventName);
                aggregateTimes.push_back(eventInfo.duration);
                aggregateCounts.push_back(1);
            } else {
                aggregateTimes[index]   = aggregateTimes[index]     + eventInfo.duration;
                aggregateCounts[index]  = aggregateCounts[index]    + 1;
            }
        }
        printf("------------------------------------------------------------\n");
        printf("          Kernel          |   Count  |   Total   | Average \n");
        printf("--------------------------+----------+-----------+----------\n");
        for (int i = 0; i < aggregateTimingNames.size(); ++i) {
            printf("--------------------------+----------+-----------+----------\n");
            printf(" %-24s |   %4d   | %8.3fms| %7.4fms\n", aggregateTimingNames[i].c_str(), aggregateCounts[i], aggregateTimes[i], aggregateTimes[i] / aggregateCounts[i]);
        }
        printf("------------------------------------------------------------\n");
    }
};

#endif