#pragma once

#include "walk/InitWalk.h"

template <typename Walker>
__global__ void pathSampler(Walker *walkers, walkId numWalker, nodeId maxStep, nodeId step, walkId id, nodeId *path) {
    const walkId tid = blockDim.x * blockIdx.x + threadIdx.x;

    for (walkId i = tid; i < numWalker; i += gridDim.x * blockDim.x) {
        nodeId source = path[(i + id + 1) * maxStep - 1];

        walkers[i].step = source == INVALID_NODE? 0: step;
        walkers[i].current = source;
        walkers[i].id = i + id;

        for (walkId j = 0; j < maxStep; j++) {
            path[(i + id) * maxStep + j] = INVALID_NODE;
        }
    }
}

template <typename Walker>
struct PathSampler: public GPUInitWalker<Walker> {
    nodeId _numNode;
    nodeId _maxStep;
    nodeId _step{0};
    nodeId *_path;

    PathSampler(walkId numWalker, nodeId maxStep, nodeId *path):
        GPUInitWalker<Walker>(numWalker), _maxStep(maxStep), _path(path)
    {}

    void setStep(nodeId step) {
        _step = step;
    }

    void init(Walker *walkers, walkId length, cudaRand &rand, CUDAStream &stream) {
        pathSampler<<<numBlock, threadPerBlock, 0, stream.get()>>>(walkers, length, _maxStep, _step, this->_generated, _path);
    }
};
