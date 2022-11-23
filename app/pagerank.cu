#include <stdio.h>

#include "LightTraffic.h"
#include "walk/NodeSampler.h"

const float alpha = 0.85;

typedef struct walker {
    nodeId current;
    nodeId step;
} Walker;

struct App { 
public:
    nodeId *_value;

    nodeId _numNode;
    walkId _numWalker;
    nodeId _maxStep;

    __device__ bool terminated(Walker &walker) {
        assert(walker.step <= _maxStep);
        return walker.step == 0;
    }

    __device__ void update(Walker &walker, const edgeId *rowptr, const nodeId *col, const nodeId nodeOffset, const edgeId edgeOffset, GPURandState &state) {
        atomicAdd(&_value[walker.current], 1);
        
        walker.step -= 1;
        
        if (terminated(walker)) {
            return;
        }

        if (curand_uniform_double(&state) > alpha) {
            walker.current = (nodeId)uniform_discrete_distribution(state, _numNode);
        }
        else {
            edgeId firstEdge = rowptr[walker.current - nodeOffset];
            edgeId deg = rowptr[walker.current + 1 - nodeOffset] - firstEdge;

            walker.current = col[firstEdge + (edgeId)uniform_discrete_distribution(state, deg) - edgeOffset];
        }
    }
};

struct AppManager {
private:
    CPUBuffer<nodeId> _h_value;
    GPUBuffer<nodeId> _d_value;
    GPUBuffer<App> _app;

    nodeId _numNode;

    int _runs;
    int _currentRun{0};

public:
    walkId _numWalker;
    nodeId _maxStep;
    GPUNodeSampler<Walker> _gpuInit;

    AppManager(nodeId numNode, Config &config, int gpuId): _h_value(numNode), _d_value(numNode, gpuId), _app(1, gpuId),
       _numNode(numNode), _runs(config._runs), _numWalker(config._numWalker), _maxStep(config._step), _gpuInit(_numWalker, numNode, _maxStep)
    {
        App app{_d_value.ptr(), _numNode, _numWalker, _maxStep};
        CPUBuffer<App> host_app(1, &app);
        host_app.to(_app);
    }

    void reduce() {
        if (_currentRun >= _runs)
            _d_value.to(_h_value);
    }

    int epochs() {
        return _runs;
    }

    void createWalkers(GPUWalkManager<App, Walker> &gpuWalkman, WalkManager<Walker> &cpuWalkman, CUDAStream &stream) {
        _currentRun++;
        if (_currentRun > _runs) {
            return;
        }

        gpuWalkman.insert(_gpuInit, cpuWalkman, stream);
    }

    bool check() {
        size_t sum = 0;
        for (nodeId i = 0; i < _numNode; i++) {
            sum += _h_value[i];
        }
        size_t expected = (size_t)(_numWalker) * (size_t)(_maxStep) * (size_t)(_runs);
        printf("sum: %zu, %zu\n", sum, expected);
        return sum == expected;
    }

    float result(nodeId i) {
        return (_h_value[i] / (_numWalker * _runs / (_numNode + 0.0))) / (_maxStep + 0.0);
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

    edgeId *rowptr = program._cpu->_graph.rowptr().ptr();
    for (nodeId i = 0, printCount = 0; i < partitions.numNode() && printCount < 20; i++) {
        edgeId deg = rowptr[i + 1] - rowptr[i];
        if (deg > 1000) {
            printf("node %u, degree %lu, value %f\n", i, deg, rw.result(i));
            printCount++;
        }
    }

    printf("node 0, degree %lu, value %f\n", rowptr[1], rw.result(0));

    bool check = rw.check();
    printf(check? "result correct\n": "result incorrect\n");

    if (!check) {
        printf("[Warning] ill-conditioned graph such as Yahoo has a very high-degree vertex whose value could exceed the max value of uint32.\n");
    }

    return 0;
}
