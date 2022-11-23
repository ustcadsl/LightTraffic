#pragma once

#include "walk/InitWalk.h"


template <typename Walker>
static __global__ void specifiedNode(Walker *walkers, walkId numWalker, nodeId node, float prob, curandState *states) {
    const walkId tid = blockDim.x * blockIdx.x + threadIdx.x;

    curandState state = states[tid];
    for (walkId i = tid; i < numWalker; i += gridDim.x * blockDim.x) {
        nodeId step = 2;
        while (curand_uniform_double(&state) > prob) {
            step += 1;
        }
        walkers[i].current = node;
        walkers[i].step = step;
    }
    state = states[tid];
}

template <typename Walker>
struct GPUSpecifiedNode: public GPUInitWalker<Walker> {
    nodeId _node;
    float _prob;

    GPUSpecifiedNode(walkId numWalker, nodeId node, float prob):
        GPUInitWalker<Walker>(numWalker), _node(node), _prob(prob)
    {}

    void init(Walker *walkers, walkId length, cudaRand &rand, CUDAStream &stream) {
        specifiedNode<<<numBlock, threadPerBlock, 0, stream.get()>>>(walkers, length, _node, _prob, rand.getRandState());
    }
};
