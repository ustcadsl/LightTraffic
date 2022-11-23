#pragma once

#include <vector>
#include <thread>

#include "partition/PartitionStrategy.h"
#include "GraphLoader.h"
#include "WalkManager.h"
#include "utils/Stopwatch.h"
#include "utils/Optional.h"
#include "Device.h"

template <typename App, typename Walker>
class GPUScheduler {

    partitionId _prevLoadedPartition{INVALID_PART};
    partitionId _prevComputePartition{INVALID_PART};

    size_t _iterations{0};
    size_t _explicit_iterations{0};

public:

    size_t iterations() {
        return _iterations;
    }

    size_t explicit_iterations() {
        return _explicit_iterations;
    }

    auto copyWalker(CPUBuffer<Walker> &walkerBatch) {
        GPUTimerGuard tg(walkerLoadingTimer);

        auto gpuBatch = gpu._walkman._active.popFreePage(gpu._copyback);
        walkerBatch.toAsync(gpuBatch, gpu._copy);

        return gpuBatch;
    }

    void computeOffMemory(GPUBuffer<Walker> &&batch, partitionId part) {
        GPUTimerGuard tg(computingTimer);

        gpu._walkman.processOffMemory(batch, part, gpu._graph.rowptr(part),  gpu._graph.col(part), gpu._compute);
    }

    void computeInMemory(partitionId part) {
        GPUTimerGuard tg(computingTimer);

        gpu._walkman.processInMemory(part, gpu._graph.rowptr(part),  gpu._graph.col(part), gpu._compute);
    }

    void evictWalker(partitionId choice) {
        GPUTimerGuard tg(walkerEvictingTimer);

        walkman.setEvictedWalker(gpu._walkman._active, choice, gpu._copyback);
    }

    void handleWalkerPoolOverflow(partitionId part) {
        gpu._compute.sync();
        copyLengthMetaData();
        
        while (gpu._walkman.hasOverflowRisk()) {
            
            auto evicted = getEvictedPartition(part);
            auto computed = getComputedPartition(evicted.value_or(INVALID_PART));
            if (evicted.has_value()) {
                evictWalker(evicted.value());
            }

            if (computed.has_value()) {
                computeInMemory(computed.value());
            }

            if (evicted.has_value() && gpu._walkman.length(evicted.value()) >= pageSize) {
                break;
            }

            gpu._compute.sync();
            copyLengthMetaData();            
        }
    }

    Optional<partitionId> getComputedPartition(partitionId except) {
        #ifndef FIFO_SCHE
        Optional<partitionId> full;
        Optional<partitionId> notfull;

        walkId min_full;
        walkId max_notfull;

        for (partitionId i = 0; i < _numPartition; i++) {
            if (i != except && gpu._graph.existed(i)) { 
                walkId numOnGPU = gpu._walkman.length(i);

                if (numOnGPU >= pageSize) {
                    walkId num = numWalkerInPartition(i);
                    if (!full.has_value() || num < min_full) {
                        full = i;
                        min_full = num;
                    }
                } else if (!full.has_value() && numOnGPU > 0) {
                    if (!notfull.has_value() || numOnGPU > max_notfull) {
                        notfull = i;
                        max_notfull = numOnGPU;
                    }
                }
            }
        }

        return full.has_value()? full: notfull;

        #else

        Optional<partitionId> choice;

        for (partitionId i = 1; i < _numPartition; i++) {
            partitionId victim = (except + i) % _numPartition;
            if (gpu._graph.existed(victim)) {
                choice = victim;
                return choice;
            }                
        }
        
        return choice;

        #endif
    }

    Optional<partitionId> getEvictedPartition(partitionId except) {
        Optional<partitionId> choice;
        walkId min;

        for (partitionId i = 0; i < _numPartition; i++) {
            if (i != except) {
                walkId num = numWalkerInPartition(i);
                if (gpu._walkman.length(i) >= pageSize) {
                    if (!choice.has_value() || num < min) {
                        choice = i;
                        min = num;
                    }
                }
            }
        }

        return choice;
    }

    partitionId findPartitionwithleastWalker(partitionId except) {
        auto partitionOnGPU = gpu._graph.existPartitions();
            
        walkId min;
        partitionId evicted;

        bool set = false;
        for (auto it = partitionOnGPU.first; it != partitionOnGPU.second; ++it) {
            partitionId i = it->first;

            if (i != except) {
                walkId num = numWalkerInPartition(i);
                if (!set || min > num) {
                    min = num;
                    evicted = i;
                    set = true;
                }
            }
        }

        assert(set);
        return evicted;
    }

    void graphLoadingPhase(partitionId part) {
        if (!gpu._graph.existed(part) && !gpu._graph.hasFreeSpace()) {
            #ifndef FIFO_SCHE
            gpu._graph.evict(findPartitionwithleastWalker(_prevLoadedPartition));
            #else
            for (partitionId i = 1; i < _numPartition; i++) {
                partitionId victim = (_prevLoadedPartition + i) % _numPartition;
                if (gpu._graph.existed(victim)) {
                    gpu._graph.evict(victim);
                    break;
                }                
            }            
            #endif
        }

        if (!gpu._graph.existed(part))
        {
            GPUTimerGuard tg(graphLoadingTimer);
            gpu._graph.loadAsync(part, cpuGraph, gpu._copy);
        }

        #ifndef NO_PIPELINE

        gpu._compute.sync();
        while (gpu._copy.isBusy()) {
            copyLengthMetaData();

            auto choice = getComputedPartition(part);
            if (choice.has_value()) {
                computeInMemory(choice.value());
                gpu._compute.sync();
            }
            else {
                break;
            }
        }

        #endif

        gpu._copy.sync();
    }

    void walkerLoadingPhase(partitionId part) {
        while (walkman.length(part) > 0) {
            auto cpuBatch = walkman.getWalkerBatch(part);
            auto gpuBatch = copyWalker(cpuBatch);

            loadedWalkerNum++;

            gpu._compute.sync();
            handleWalkerPoolOverflow(part);

            gpu._copy.sync();

            computeOffMemory(std::move(gpuBatch), part);

            walkman.insertFreePage(cpuBatch);
        }

        gpu._compute.sync();
    }

    void walkerComputingPhase(partitionId part) {
        int numBatch = gpu._walkman.numBatch(part);

        GPUTimerGuard tg(computingTimer);

        gpu._walkman.processInMemory(part, gpu._graph.rowptr(part),  gpu._graph.col(part), gpu._compute, numBatch);
    }

    bool shouldZerocopy(partitionId part) {
        
        if (!gpu._graph.existed(part) && numWalkerInPartition(part) < zerocopy_threshold) {
            gpu._compute.sync();
            copyLengthMetaData();
            return numWalkerInPartition(part) < zerocopy_threshold;
        }

        return false;
    }

    void zeroCopyCompute(partitionId part) {
        GPUTimerGuard tg(zerocopyTimer);
        int numBatch = gpu._walkman.numBatch(part);            
        gpu._walkman.processInMemory(part, cpuGraph.rowptr(part)->ptr(), cpuGraph.col(part)->ptr(), gpu._compute, numBatch);
        
        while (walkman.length(part) > 0) {
            auto cpuBatch = walkman.getWalkerBatch(part);
            auto gpuBatch = copyWalker(cpuBatch);

            gpu._compute.sync();
            handleWalkerPoolOverflow(part);

            gpu._copy.sync();

            GPUTimerGuard tg(zerocopyTimer);
            gpu._walkman.processOffMemory(gpuBatch, part, cpuGraph.rowptr(part)->ptr(), cpuGraph.col(part)->ptr(), gpu._compute);

            walkman.insertFreePage(cpuBatch);
        }
    }

    int evictedWalkerNum = 0;
    int loadedWalkerNum = 0;

    GPU<App, Walker> &gpu;
    CPUGraph &cpuGraph;
    WalkManager<Walker> &walkman;
    partitionId _numPartition;
    size_t zerocopy_threshold;

    GPUTimer graphLoadingTimer;
    GPUTimer computingTimer;
    GPUTimer zerocopyTimer;
    GPUTimer walkerLoadingTimer;
    GPUTimer walkerEvictingTimer;

    GPUScheduler(PartitionStrategy &partitions, CPU<Walker> &cpu, GPU<App, Walker> &gpu, int mainGPU = 0):
        gpu(gpu),
        cpuGraph(cpu._graph),
        walkman(cpu._walkman),
        _numPartition(partitions.numPartition()),
        zerocopy_threshold(partitions.partitionSize() / cachelineSize / 2),
        graphLoadingTimer(gpu._copy),
        computingTimer(gpu._compute),
        zerocopyTimer(gpu._compute),
        walkerLoadingTimer(gpu._copy),
        walkerEvictingTimer(gpu._copyback)
    {
        #ifdef D_ZEROCOPY_THRESHOLD
            zerocopy_threshold = D_ZEROCOPY_THRESHOLD;
        #endif

        #ifdef D_ZEROCOPY_THRESHOLD_INF
            zerocopy_threshold = 0xFFFFFFFFFFFFFFFF;
        #endif
    }

    void run(partitionId part) {
        _prevComputePartition = part;

        _iterations++;

        if (shouldZerocopy(part)) {
            zeroCopyCompute(part);
            return;
        }

        _explicit_iterations++;

        graphLoadingPhase(part);
        _prevLoadedPartition = part;
        
        walkerLoadingPhase(part);
        copyLengthMetaData();
        
        walkerComputingPhase(part);
    }

    walkId numWalkerInPartition(partitionId p) {
        walkId result = walkman.length(p);
        result += gpu._walkman.length(p);

        return result;
    }

    void copyLengthMetaData() {        
        gpu._walkman._active.copyLengthMetaData(gpu._copyback);
        gpu._copyback.sync();
    }

    partitionId previousComputedPartition() {
        return _prevComputePartition;
    }
};

template <typename App, typename Walker>
class Scheduler {
private:
    GPUScheduler<App, Walker> scheduler;

    PartitionStrategy &_partitions;
    CPU<Walker> &_cpu;
    GPU<App, Walker> &_gpu;

    bool _start{false};

    partitionId _numPartition;

    Timer timer;

    bool _sync{false};

    #ifndef FIFO_SCHE

    bool nextPartition(partitionId &choice) {
        scheduler.copyLengthMetaData();
        
        walkId max = 0;
        bool picked = false;

        for (partitionId i = 0; i < _numPartition; i++) {
            if (i == scheduler.previousComputedPartition() && !_sync) {
                continue;
            }

            walkId num = scheduler.numWalkerInPartition(i);
            if (max < num) {
                max = num;
                choice = i;
                picked = true;
            }
        }
        
        return picked;
    }

    
    #else

    bool nextPartition(partitionId &choice) {
        static partitionId p = 0;
        if (p == 0) {
            scheduler.copyLengthMetaData();
            bool end = true;
            for (partitionId i = 0; i < _partitions.numPartition(); i++) {
                if (scheduler.numWalkerInPartition(i) > 0) {
                    end = false;
                } 
            }

            if (end) return false;
        }

        while (p < _partitions.numPartition()) {
            choice = p++;
            if (scheduler.numWalkerInPartition(choice) > 0) {
                return true;
            }
        }

        p = 0;
        return false;
    }

    #endif

public:
    Scheduler(PartitionStrategy &partitions, CPU<Walker> &cpu, GPU<App, Walker> &gpu):
        scheduler(partitions, cpu, gpu, 0), _partitions(partitions), _cpu(cpu), _gpu(gpu), _numPartition(partitions.numPartition())
    {}

    void start() {
        TimerGuard tg(timer);
        partitionId choice;

        while (nextPartition(choice)) {
            _sync = false;
            do {
                scheduler.run(choice);
            } while (nextPartition(choice));
            cudaDeviceSynchronize();
            _sync = true;

        }
    }

    void timing() {
        printf("Running Time: %f ms\n", timer.elapsed());
        printf("iterations: %lu, explicit: %lu\n", scheduler.iterations(), scheduler.explicit_iterations());

        #ifdef NO_EVENT_TIMER
        printf("Warning: GPU timer is disabled, the following statistics is wrong!\n");
        #endif

        printf("graph loading time: %f ms, calls: %lu\n", scheduler.graphLoadingTimer.elapsed(), scheduler.graphLoadingTimer.calls());
        printf("computing time: %f ms, calls: %lu\n", scheduler.computingTimer.elapsed(), scheduler.computingTimer.calls());
        printf("zero copy time: %f ms, calls: %lu\n", scheduler.zerocopyTimer.elapsed(), scheduler.zerocopyTimer.calls());
        printf("walker loading time: %f ms, calls: %lu\n", scheduler.walkerLoadingTimer.elapsed(), scheduler.walkerLoadingTimer.calls());
        printf("walker evicting time: %f ms, calls: %lu\n", scheduler.walkerEvictingTimer.elapsed(), scheduler.walkerEvictingTimer.calls());
    }
};
