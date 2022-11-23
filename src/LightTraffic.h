#pragma once

#include "Config.h"
#include "partition/PartitionStrategy.h"
#include "Type.h"
#include "GraphLoader.h"
#include "WalkManager.h"
#include "Scheduler.h"
#include "Device.h"

template <typename AppManager, typename App, typename Walker>
class LightTraffic {
public:
    PartitionStrategy &_partitions;
    DiskGraph _diskGraph;
    CPU<Walker> *_cpu;
    GPU<App, Walker> *_gpu;

    LightTraffic(Config &config, AppManager &appManager):
        _partitions(*(config._partitions)),
        _diskGraph(config._file)
    {
        _cpu = new CPU<Walker>(config);
        _gpu = new GPU<App, Walker>(config, appManager.GPUApp(), 0);

        printf("number of partitons: %u, graph memory pool: %lu MB, walker memory pool: %lu MB\n",
            _partitions.numPartition(),
            _partitions.partitionSize() * config._numDevicePartition / 1024 / 1024,
            pageSize * _gpu->_walkman.numPage() * sizeof(Walker) / 1024 / 1024
        );
    }

    void start(AppManager &appManager) {
        _cpu->_graph.load(_diskGraph);
        Scheduler<App, Walker> scheduler(_partitions, *_cpu, *_gpu);

        for (int i = 0; i < appManager.epochs(); i++) {
            appManager.createWalkers(_gpu->_walkman, _cpu->_walkman, _gpu->_compute);
            scheduler.start();
            appManager.reduce();
        }

        scheduler.timing();
    }
};
