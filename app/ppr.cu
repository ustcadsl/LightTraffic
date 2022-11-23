#include <stdio.h>

#include "LightTraffic.h"
#include "walk/SingleSource.h"

typedef struct walker {
    nodeId current;
    nodeId step;
} Walker;

struct App { 
public:
    nodeId *_value;

    nodeId _numNode;
    walkId _numWalker;

    __device__ bool terminated(Walker &walker) {
        return walker.step == 0;
    }

    __device__ void update(Walker &walker, const edgeId *rowptr, const nodeId *col, const nodeId nodeOffset, const edgeId edgeOffset, GPURandState &state) {
        if (walker.step == 1) {
            atomicAdd(&_value[walker.current], 1);
        }

        walker.step -= 1;
        
        if (terminated(walker)) {
            return;
        }
        
        edgeId firstEdge = rowptr[walker.current - nodeOffset];
        edgeId deg = rowptr[walker.current + 1 - nodeOffset] - firstEdge;

        if (deg > 0)
            walker.current = col[firstEdge + (edgeId)uniform_discrete_distribution(state, deg) - edgeOffset];
        else {
            atomicAdd(&_value[walker.current], 1);
            walker.step = 0;
        }
    }
};

struct AppManager {
private:
    CPUBuffer<nodeId> _h_value;
    GPUBuffer<nodeId> _d_value;
    GPUBuffer<App> _app;

    nodeId _numNode;

public:
    walkId _numWalker;
    GPUSpecifiedNode<Walker> _gpuInit;

    AppManager(nodeId numNode, Config &config, int gpuId): _h_value(numNode), _d_value(numNode, gpuId), _app(1, gpuId),
       _numNode(numNode), _numWalker(config._numWalker), _gpuInit(_numWalker, config._source, config._prob)
    {
        App app{_d_value.ptr(), _numNode, _numWalker};
        CPUBuffer<App> host_app(1, &app);
        host_app.to(_app);
    }

    void reduce() {
        _d_value.to(_h_value);
    }

    int epochs() {
        return 1;
    }

    void createWalkers(GPUWalkManager<App, Walker> &gpuWalkman, WalkManager<Walker> &cpuWalkman, CUDAStream &stream) {
        gpuWalkman.insert(_gpuInit, cpuWalkman, stream);
    }

    bool check() {
        size_t sum = 0;
        for (nodeId i = 0; i < _numNode; i++) {
            sum += _h_value[i];
        }
        size_t expected = _numWalker;
        printf("sum: %lu, %lu\n", sum, expected);
        return sum == expected;
    }

    float result(nodeId i) {
        return (_h_value[i] / (_numWalker / (_numNode + 0.0)));
    }

    auto GPUApp() {
        return _app.ptr();
    }
};

int main(int argc, char* argv[])
{   
    Config config(argc, argv);
    PartitionStrategy &partitions = *(config._partitions);

    int gpuId = 0;
    AppManager rw(partitions.numNode(), config, gpuId);

    LightTraffic<AppManager, App, Walker> program(config, rw);
    program.start(rw);

    nodeId target = 2;
    printf("PageRank value of node %u: %f\n", target, rw.result(target));

    printf(rw.check()? "result correct\n": "result incorrect\n");

    return 0;
}
