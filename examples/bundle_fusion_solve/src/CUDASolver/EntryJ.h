#pragma once
#include <cuda_runtime.h>
//correspondence_idx -> image_Idx_i,j
struct EntryJ {
    unsigned int imgIdx_i;
    unsigned int imgIdx_j;
    float3 pos_i;
    float3 pos_j;

    __host__ __device__
        void setInvalid() {
        imgIdx_i = (unsigned int)-1;
        imgIdx_j = (unsigned int)-1;
    }
    __host__ __device__
        bool isValid() const {
        return imgIdx_i != (unsigned int)-1;
    }
};