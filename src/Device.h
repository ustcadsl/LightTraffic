#pragma once

#include "Config.h"
#include "GraphLoader.h"
#include "WalkManager.h"

template <typename Walker>
class CPU {
public:
    CPUGraph _graph;
    WalkManager<Walker> _walkman;

    CPU(Config &config):
        _graph(config._partitions->numPartition(), *(config._partitions)),
        _walkman(config._numWalker - config._numDeviceWalker, *(config._partitions))
    {
        _graph.cudaRegister();
        _walkman.cudaRegister();
    }
};

template <typename App, typename Walker>
class GPU {

public:
    int _id;
    App *_app;
    GPUGraph _graph;
    GPUWalkManager<App, Walker> _walkman;

    CUDAStream _copy;
    CUDAStream _compute;
    CUDAStream _copyback;

    GPU(Config &config, App *app, int id):
        _id(id),
        _app(app),
        _graph(config._numDevicePartition, *(config._partitions), id),
        _walkman(config._numDeviceWalker, *(config._partitions), _app, id),
        _copy(id),
        _compute(id),
        _copyback(id)
    {}

    void setDevice()
    {
        cudaSetDevice(_id);
    }

    void viewProperties()
    {
        setDevice();
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, _id);
        
        printf("wrapSize: %d, totalGlobalMem: %lu, maxThreadsPerBlock: %d, SMCount: %d, sharedMemPerBlock: %d\n", \
            prop.warpSize, prop.totalGlobalMem, prop.maxThreadsPerBlock, prop.multiProcessorCount, prop.sharedMemPerBlock);
    }
};
