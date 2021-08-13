#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
// Enable run time assertion checking in kernel code
#define cudaAssert(condition) if (!(condition)) { printf("ASSERT: %s %s\n", #condition, __FILE__); }

#define cudaSafeCall(err)  _internal_cudaSafeCall(err,__FILE__,__LINE__)

// Adapted from the G3D innovation engine's debugAssert.h
#    if defined(_MSC_VER) 
#       define rawBreak()  __debugbreak();
#    elif defined(__i386__)
// gcc on intel
#       define rawBreak() __asm__ __volatile__ ( "int $3" ); 
#    else
// some other gcc
#      define rawBreak() ::abort()
#   endif

inline void _internal_cudaSafeCall(cudaError err, const char *file, const int line){
    if (cudaSuccess != err) {
        printf("%s(%i) : cudaSafeCall() error: %s\n", file, line, cudaGetErrorString(err));
        rawBreak();
    }
}

#include <assert.h>
#include <vector>
template <class T>
class CudaArray {
public:
    void update(const T* newData, size_t count) {
        destructiveResize(count);
        cudaSafeCall(cudaMemcpy(m_data, newData, sizeof(T)*m_size, cudaMemcpyHostToDevice));
    }

    void update(const std::vector<T>& newData) {
        update(newData.data(), newData.size());
    }

    void readBack(T* cpuBuffer, size_t count) {
        cudaSafeCall(cudaMemcpy(cpuBuffer, m_data, sizeof(T)*std::min(count,m_size), cudaMemcpyDeviceToHost));
    }

    void readBack(std::vector<T>& cpuBuffer) {
        cpuBuffer.resize(m_size);
        readBack(cpuBuffer.data(), m_size);
    }

    void destructiveResize(size_t count) {
        if (count > m_allocated) {
            if (m_data) {
                cudaSafeCall(cudaFree(m_data));
            }
            m_allocated = count;
            cudaSafeCall(cudaMalloc(&m_data, sizeof(T)*count));
        }
        m_size = count;
    }

    void alloc(size_t count) {
        assert(m_size == 0);
        destructiveResize(count);
    }

    size_t size() const {
        return m_size;
    }

    T* data() const {
        return m_data;
    }

    CudaArray() {}

    ~CudaArray() {
        if (m_data) {
            cudaSafeCall(cudaFree(m_data));
        }
    }

protected:
    size_t m_allocated = 0;
    size_t m_size = 0;
    T* m_data = nullptr;
};