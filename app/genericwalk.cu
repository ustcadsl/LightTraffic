#include <stdio.h>

#include "LightTraffic.h"
#include "walk/NodeSampler.h"

typedef struct walker {
    nodeId current;
    nodeId step;
    uint32_t id;
} Walker;

struct App { 
public:
    nodeId _numNode;
    nodeId _maxStep;

    __device__ bool terminated(Walker &walker) {
        assert(walker.step <= _maxStep);
        return walker.step == 0;
    }

    __device__ void update(Walker &walker, const edgeId *rowptr, const nodeId *col, const nodeId nodeOffset, const edgeId edgeOffset, GPURandState &state) {        
        edgeId firstEdge = rowptr[walker.current - nodeOffset];
        edgeId deg = rowptr[walker.current + 1 - nodeOffset] - firstEdge;

        if (deg > 0)
            walker.current = col[firstEdge + (edgeId)uniform_discrete_distribution(state, deg) - edgeOffset];
        else
            walker.step = 1;

        walker.step -= 1;
    }
};

struct AppManager {
private:
    nodeId _numNode;
    walkId _numWalker;
    nodeId _pathLength;

    GPUBuffer<App> _app;

    GPUNodeSampler<Walker> _nodeSampler;

    nodeId _beginStep{0};
    nodeId _endStep{0};

    int _runs;
    int _currentRun{0};

public:

    AppManager(nodeId numNode, Config &config, int gpuId):
        _numNode(numNode), _numWalker(config._numWalker), _pathLength(config._step),
        _app(1, gpuId),
        _nodeSampler(_numWalker, numNode, _pathLength),
        _runs(config._runs)
    {
        App app{_numNode, _pathLength};
        CPUBuffer<App> host_app(1, &app);
        host_app.to(_app);
    }

    void reduce() {
        return;
    }

    int epochs() {
        return _runs;
    }

    void createWalkers(GPUWalkManager<App, Walker> &gpuWalkman, WalkManager<Walker> &cpuWalkman, CUDAStream &stream) {
        _currentRun++;
        if (_currentRun > _runs) {
            return;
        }

        gpuWalkman.insert(_nodeSampler, cpuWalkman, stream);
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

    return 0;
}
