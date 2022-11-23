#pragma once

#include "walk/InitWalk.h"

template <typename Walker>
static __global__ void sampleNode(Walker *walkers, walkId numWalker, nodeId numNode, nodeId maxStep, curandState *states) {
    const walkId tid = blockDim.x * blockIdx.x + threadIdx.x;

    curandState state = states[tid];
    for (walkId i = tid; i < numWalker; i += gridDim.x * blockDim.x) {
        nodeId source = uniform_discrete_distribution(state, numNode);
        walkers[i] = Walker{source, maxStep};
    }
    state = states[tid];
}

template <typename Walker>
struct GPUNodeSampler: public GPUInitWalker<Walker> {
    nodeId _numNode;
    nodeId _maxStep;

    GPUNodeSampler(walkId numWalker, nodeId numNode, nodeId maxStep):
        GPUInitWalker<Walker>(numWalker), _numNode(numNode), _maxStep(maxStep)
    {}

    void init(Walker *walkers, walkId length, cudaRand &rand, CUDAStream &stream) {
        sampleNode<Walker><<<numBlock, threadPerBlock, 0, stream.get()>>>(walkers, length, _numNode, _maxStep, rand.getRandState());
    }
};

template <typename Walker>
__global__ void sampleNodeWithId(Walker *walkers, walkId numWalker, nodeId numNode, nodeId maxStep, walkId id, nodeId *path, curandState *states) {
    const walkId tid = blockDim.x * blockIdx.x + threadIdx.x;

    curandState state = states[tid];
    for (walkId i = tid; i < numWalker; i += gridDim.x * blockDim.x) {
        nodeId source = uniform_discrete_distribution(state, numNode);

        walkers[i].step = maxStep - 1;
        walkers[i].current = source;
        walkers[i].id = i + id;

        for (walkId j = 1; j < maxStep; j ++) {
            path[(i + id) * maxStep + j] = INVALID_NODE;
        }
        path[(i + id) * maxStep] = source;
    }
    state = states[tid];
}

template <typename Walker>
struct GPUNodeSamplerWithId: public GPUInitWalker<Walker> {
    nodeId _numNode;
    nodeId _maxStep;
    nodeId *_path;

    GPUNodeSamplerWithId(walkId numWalker, nodeId numNode, nodeId maxStep, nodeId *path):
        GPUInitWalker<Walker>(numWalker), _numNode(numNode), _maxStep(maxStep), _path(path)
    {}

    void init(Walker *walkers, walkId length, cudaRand &rand, CUDAStream &stream) {
        sampleNodeWithId<<<numBlock, threadPerBlock, 0, stream.get()>>>(walkers, length, _numNode, _maxStep, this->_generated, _path, rand.getRandState());
    }
};
