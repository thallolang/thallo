#pragma once

#ifndef _SOLVER_Stereo_UTIL_
#define _SOLVER_Stereo_UTIL_

#include "SolverUtil.h"

#include <cutil_inline.h>
#include <cutil_math.h>

#define THREADS_PER_BLOCK 512 // keep consistent with the CPU
#define WARP_SIZE 32

#define AVOID_UNDEFINED_WARP_INSTRUCTIONS 1


template <typename F>
__device__ __inline__ F reduce_peers(uint peers, F &x) {
    int lane = threadIdx.x & 31;

    // find the peer with lowest lane index
    int first = __ffs(peers) - 1;

    // calculate own relative position among peers
    int rel_pos = __popc(peers << (32 - lane));

    // ignore peers with lower (or same) lane index
    peers &= (0xfffffffe << lane);

    while (__any(peers)) {
        // find next-highest remaining peer
        int next = __ffs(peers);

        // __shfl() only works if both threads participate, so we always do.
        F t = __shfl(x, next - 1);

        // only add if there was anything to add
        if (next) x += t;

        // all lanes with their least significant index bit set are done
        uint done = rel_pos & 1;

        // remove all peers that are already done
        peers &= ~__ballot(done);

        // abuse relative position as iteration counter
        rel_pos >>= 1;
    }

    // distribute final result to all peers (optional)
    F res = __shfl(x, first);

    return res;
}

template <typename F>
__device__ __inline__ void atomic_reduce_peers(F* dest, uint peers, F &x) {
    int lane = threadIdx.x & 31;

    // find the peer with lowest lane index
    int first = __ffs(peers) - 1;

    // calculate own relative position among peers
    int rel_pos = __popc(peers << (32 - lane));

    // ignore peers with lower (or same) lane index
    peers &= (0xfffffffe << lane);

    while (__any(peers)) {
        // find next-highest remaining peer
        int next = __ffs(peers);

        // __shfl() only works if both threads participate, so we always do.
        F t = __shfl(x, next - 1);

        // only add if there was anything to add
        if (next) x += t;

        // all lanes with their least significant index bit set are done
        uint done = rel_pos & 1;

        // remove all peers that are already done
        peers &= ~__ballot(done);

        // abuse relative position as iteration counter
        rel_pos >>= 1;
    }

    if (lane == first) { // only leader threads for each key perform atomics
        atomicAdd(dest, x);
    }
}

template <typename F>
__inline__ __device__ void warpReduceAndAtomicAdd(F* dest, F val) {
    uint peers = __ballot(1);
    return atomic_reduce_peers<F>(dest, peers, val);
}

template <typename F>
__inline__ __device__ F warpReduce(F val) {
    int offset = 32 >> 1;
    while (offset > 0) {
        val = val + __shfl_down(val, offset, 32);
        offset = offset >> 1;
    }
    return val;
}

#endif
