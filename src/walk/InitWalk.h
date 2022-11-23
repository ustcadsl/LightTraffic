#pragma once

#include "Type.h"
#include "buffer/Buffer.h"
#include "Rand.h"

template <typename Walker>
struct GPUInitWalker {
protected:
    walkId _numWalker;
    walkId _generated{0};

public:
    GPUInitWalker(walkId numWalker): _numWalker(numWalker) {}

    void create(Walker *walkers, walkId &length, cudaRand &rand, CUDAStream &stream) {
        length = length > _numWalker - _generated? _numWalker - _generated: length;
        init(walkers, length, rand, stream);
        _generated += length;
    }

    bool done() {
        return _generated == _numWalker;
    }

    walkId numWalker() {
        return _numWalker;
    }

    walkId generated() {
        return _generated;
    }

    void reset() {
        _generated = 0;
    }

    virtual void init(Walker *walkers, walkId length, cudaRand &rand, CUDAStream &stream) = 0;
    virtual ~GPUInitWalker(){}
};
