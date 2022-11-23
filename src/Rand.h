#pragma once

#include <chrono>
#include <random>
#include <omp.h>

#include <curand_kernel.h>

using GPURandState = curandState;

static __global__ void initCudaRand(GPURandState *states, unsigned long seed)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &states[tid]);
}

__device__ u_int64_t uniform_discrete_distribution(GPURandState &state, nodeId n) {
    return ((static_cast<u_int64_t>(curand(&state)) << 32) + curand(&state)) % n;
}

class cudaRand
{
private:
    GPUBuffer<GPURandState> _states;

public:
    cudaRand(int block, int threadPerBlock, int gpuId): _states(block * threadPerBlock, gpuId)
    {
        std::chrono::nanoseconds time;
        auto seed = time.count();
        initCudaRand<<<block, threadPerBlock>>>(_states.ptr(), seed);
    }

    GPURandState* getRandState() {
        return _states.ptr();
    }
};
