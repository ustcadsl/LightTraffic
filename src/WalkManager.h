#pragma once

#include <memory>

#include "Type.h"
#include "partition/PartitionStrategy.h"
#include "buffer/Pool.h"
#include "walk/InitWalk.h"
#include "Rand.h"

template <typename Walker>
class WalkManager {
private:
    PartitionStrategy &_partitions;
    partitionId _numPartition;
    ListPool<Walker> _active;
    
public:

    WalkManager(walkId numWalker, PartitionStrategy &partitions):
        _partitions(partitions), _numPartition(partitions.numPartition()),
        _active(_numPartition, pageSize, numWalker)
    {}

    auto getWalkerBatch(partitionId i) {
        return _active.removePage(i);
    }

    void insertFreePage(CPUBuffer<Walker> &buffer) {
        _active.insertFreePage(buffer);
    }

    walkId length(partitionId i) {
        return _active.length(i);
    }

    walkId numWalker() {
        walkId sum = 0;
        for (partitionId i = 0; i < _numPartition; i++) {
            sum += length(i);
        }
        return sum;
    }

    void clear(partitionId i) {
        _active.clear(i);
    }

    void cudaRegister() {
        _active.cudaRegister();
    }

    void setEvictedWalker(GPUListPool<Walker> &gpuWalkerPool, partitionId i, CUDAStream &stream) {
        auto batch = _active.getFreePage();
        gpuWalkerPool.evict(i, stream, batch);
        _active.insertPage(i, batch);
    }
};

static __device__ void clearGlobalOffset(u_int32_t *globalOffset, const partitionId numPartition) {
    for (partitionId i = threadIdx.x; i < numPartition; i += blockDim.x) {
        globalOffset[i] = 0;
    }
    __syncthreads();
}

template <typename Walker>
static __device__ void insertToLocalPool(const u_int32_t index, const Walker &walker, u_int32_t *globalOffset, u_int32_t *localOffset, partitionId *inPartition, DevicePartition *devicePartition) {
    partitionId part = devicePartition->findPartition(walker.current);
    localOffset[index] = atomicAdd(&globalOffset[part], 1);
    inPartition[index] = part;
}

static __device__ void getPrefix(partitionId numPartition, const u_int32_t *globalOffset, u_int32_t *prefix) {
    __shared__ typename BlockScan::TempStorage temp_storage;

    if (threadIdx.x == 0) {
        prefix[0] = 0;
    }

    u_int32_t count = 0;
    
    for (partitionId iter = 0; iter * blockDim.x < numPartition; iter++) {
        partitionId i = iter * blockDim.x + threadIdx.x;
        count += i < numPartition? globalOffset[i]: 0;

        BlockScan(temp_storage).InclusiveSum(count, count);

        if (i < numPartition) {
            prefix[i + 1] = count;
        }
        __syncthreads();

        count = threadIdx.x == 0 && i + blockDim.x < numPartition? prefix[i + blockDim.x]: 0;
    }
}

#ifdef NO_RANDOM_ACCESS_REDUCE

template <typename Walker>
static __device__ void insertToGlobalPool(const u_int32_t tid, DeviceListPool<Walker> *pool, const Walker *walkers, const u_int32_t numWalker, u_int32_t *globalOffset, partitionId numPartition, u_int32_t *localOffset, partitionId *inPartition) {    
    __syncthreads();

    u_int32_t pageCapacity = pool->_pageCapacity;

    for (partitionId i = threadIdx.x; i < numPartition; i += blockDim.x) {
        globalOffset[i] = atomicAdd((u_int32_t *)&(pool->_length[i]), globalOffset[i]);
    }
    __syncthreads();

    for (u_int32_t i = 0; i * gridDim.x * blockDim.x + tid < numWalker; i++) {
        if (localOffset[i] != INVALID_WALK) {
            partitionId part = inPartition[i];
            walkId index = globalOffset[part] + localOffset[i];
            pool->getRear(part, index / pageCapacity)[index % pageCapacity] = walkers[i * gridDim.x * blockDim.x + tid];
        }
    }
}

#else

template <typename Walker>
static __device__ void insertToGlobalPool(const u_int32_t tid, DeviceListPool<Walker> *pool, const Walker *walkers, const u_int32_t numWalker, u_int32_t *globalOffset, partitionId numPartition, u_int32_t *localOffset, partitionId *inPartition) {
    __syncthreads();
    
    u_int32_t *globalOffsetPrefix = (u_int32_t *)(globalOffset + numPartition);
    getPrefix(numPartition, globalOffset, globalOffsetPrefix);

    u_int32_t *shuffledIndex = (u_int32_t *)(globalOffsetPrefix + numPartition + 1);
    u_int32_t *shuffledGlobalOffset = (u_int32_t *)(shuffledIndex + walkerPerThread * threadPerBlock);
    partitionId *shuffledPartition = (partitionId *)(shuffledGlobalOffset + walkerPerThread * threadPerBlock);

    __syncthreads();

    u_int32_t pageCapacity = pool->_pageCapacity;
    u_int32_t numInserted = globalOffsetPrefix[numPartition];

    for (partitionId i = threadIdx.x; i < numPartition; i += blockDim.x) {
        globalOffset[i] = atomicAdd((u_int32_t *)&(pool->_length[i]), globalOffset[i]);
    }
    __syncthreads();

    for (u_int32_t i = 0; i * gridDim.x * blockDim.x + tid < numWalker; i++) {
        if (localOffset[i] != INVALID_WALK) {
            partitionId part = inPartition[i];
            u_int32_t localIndex = globalOffsetPrefix[part] + localOffset[i];

            shuffledIndex[localIndex] = i * gridDim.x * blockDim.x + tid;
            shuffledPartition[localIndex] = part;
            shuffledGlobalOffset[localIndex] = globalOffset[part] + localOffset[i];
        }
    }
    __syncthreads();
    
    for (u_int32_t i = threadIdx.x; i < numInserted; i += blockDim.x) {
        partitionId part = shuffledPartition[i];
        u_int32_t index = shuffledGlobalOffset[i];

        pool->getRear(part, index / pageCapacity)[index % pageCapacity] = walkers[shuffledIndex[i]];
    }
}

#endif

#ifndef ONE_LEVEL_QUEUE

template <typename Walker>
static __device__ void insertToQueueKernel(DeviceListPool<Walker> *pool, Walker *walkers, u_int32_t numWalker, const partitionId numPartition, DevicePartition *devicePartition)
{
    const u_int32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ u_int8_t tiled[sharedMemPerBlock - sizeof(BlockScan::TempStorage)];

    u_int32_t *globalOffset = (u_int32_t *)tiled;
    u_int32_t localOffset[walkerPerThread];
    partitionId inPartition[walkerPerThread];

    clearGlobalOffset(globalOffset, numPartition);

    for (u_int32_t i = 0; i * gridDim.x * blockDim.x + tid < numWalker; i++) {
        if (walkers[i * gridDim.x * blockDim.x + tid].step != 0) {
            insertToLocalPool(i, walkers[i * gridDim.x * blockDim.x + tid], globalOffset, localOffset, inPartition, devicePartition);
        }
        else {
            localOffset[i] = INVALID_WALK;
        }
    }
  
    insertToGlobalPool(tid, pool, walkers, numWalker, globalOffset, numPartition, localOffset, inPartition);
}

#else


template <typename Walker>
static __device__ void insertToQueueKernel(DeviceListPool<Walker> *pool, Walker *walkers, u_int32_t numWalker, const partitionId numPartition, DevicePartition *devicePartition)
{
    const u_int32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    u_int32_t pageCapacity = pool->_pageCapacity;

    for (u_int32_t i = 0; i * gridDim.x * blockDim.x + tid < numWalker; i++) {
        if (walkers[i * gridDim.x * blockDim.x + tid].step != 0) {
            partitionId part = devicePartition->findPartition(walkers[i * gridDim.x * blockDim.x + tid].current);
            u_int32_t index = atomicAdd((u_int32_t *)&(pool->_length[part]), 1);

            pool->getRear(part, index / pageCapacity)[index % pageCapacity] = walkers[i * gridDim.x * blockDim.x + tid];
        }
    }
}

#endif

template <typename Walker>
static __global__ void insertReservedToQueue(DeviceListPool<Walker> *pool, const partitionId partition, const partitionId numPartition, DevicePartition *devicePartition)
{

    Walker *walkers = pool->getHead(partition);
    u_int32_t numWalker = pool->headPageLength(partition);

    insertToQueueKernel(pool, walkers, numWalker, numPartition, devicePartition);
}

template <typename Walker>
static __global__ void insertPushedToQueue(DeviceListPool<Walker> *pool, Walker *walkers, u_int32_t numWalker, const partitionId numPartition, DevicePartition *devicePartition)
{
    insertToQueueKernel(pool, walkers, numWalker, numPartition, devicePartition);
}

template <typename App, typename Walker>
static __device__ void randomWalkKernel(Walker *walkers, const u_int32_t numWalker, const nodeId nodeBegin, const nodeId nodeEnd, const edgeId edgeBegin, const edgeId *rowptr, const nodeId *col, GPURandState *states,\
    DeviceListPool<Walker> *pool, DevicePartition *devicePartition, App *rw)
{
    const u_int32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    GPURandState state = states[tid];

    for (u_int32_t i = 0; i * gridDim.x * blockDim.x + tid < numWalker; i++) {
        Walker walker = walkers[i * gridDim.x * blockDim.x + tid];
        
        while (walker.step > 0 && walker.current >= nodeBegin && walker.current < nodeEnd)
            rw->update(walker, rowptr, col, nodeBegin, edgeBegin, state);

        walkers[i * gridDim.x * blockDim.x + tid] = walker;
    }

    states[tid] = state;
}

template <typename App, typename Walker>
static __global__ void randomWalkInsidePool(const partitionId partition, const nodeId nodeBegin, const nodeId nodeEnd, const edgeId edgeBegin, const edgeId *rowptr, const nodeId *col, GPURandState *states,\
    DeviceListPool<Walker> *pool, DevicePartition *devicePartition, App *rw)
{
    Walker *walkers = pool->getHead(partition);
    u_int32_t numWalker = pool->headPageLength(partition);

    randomWalkKernel<App>(walkers, numWalker, nodeBegin, nodeEnd, edgeBegin, rowptr, col, states, pool, devicePartition, rw);
}

template <typename App, typename Walker>
static __global__ void randomWalkInsidePoolAllBatch(u_int32_t times, const partitionId partition, const nodeId nodeBegin, const nodeId nodeEnd, const edgeId edgeBegin, const edgeId *rowptr, const nodeId *col, GPURandState *states,\
    DeviceListPool<Walker> *pool, DevicePartition *devicePartition, App *rw)
{
    u_int32_t listCapacity = pool->_listCapacity;
    u_int32_t batch = pool->_head[partition];

    for (u_int32_t i = 0; i < times; i++) {
        Walker *walkers = pool->getPageAddr(partition, batch);
        bool isLastBatch = batch == pool->_rear[partition];
        u_int32_t numWalker = isLastBatch? pool->_length[partition]: pool->_pageCapacity;

        randomWalkKernel<App>(walkers, numWalker, nodeBegin, nodeEnd, edgeBegin, rowptr, col, states, pool, devicePartition, rw);

        batch = (batch + 1) % listCapacity;
    }
}

template <typename App, typename Walker>
static __global__ void randomWalkOutsidePool(Walker *walkers, const u_int32_t numWalker, const nodeId nodeBegin, const nodeId nodeEnd, const edgeId edgeBegin, const edgeId *rowptr, const nodeId *col, GPURandState *states,\
    DeviceListPool<Walker> *pool, DevicePartition *devicePartition, App *rw)
{
    const partitionId numPartition = devicePartition->numPartition();
    randomWalkKernel<App>(walkers, numWalker, nodeBegin, nodeEnd, edgeBegin, rowptr, col, states, pool, devicePartition, rw);
}

template <typename App, typename Walker>
class GPUWalkManager {
private:
    PartitionStrategy &_partitions;
    partitionId _numPartition;
    DevicePartition *_devicePartitions;
    size_t _numPage;
    walkId _maxWalker;
    cudaRand _rand;
    App *_app;

    int _gpuId;

    partitionId getOutMemPartition(partitionId numPartition) {
        partitionId choice;
        walkId min;
        bool set = false;

        for (partitionId i = 0; i < numPartition; i++) {
            if (length(i) > pageSize) {
                if (!set || min > length(i)) {
                    choice = i;
                    min = length(i);
                    set = true;
                }
            }
        }
        return choice;
    }

public:
    GPUListPool<Walker> _active;

    size_t numPageRequired(walkId numWalker) {
        return (numWalker == 0? 0: (numWalker - 1) / pageSize + 3) + _numPartition * 2;
    }

    bool hasOverflowRisk() {
        return numPageRequired(numWalker()) > numPage();
    }

    GPUWalkManager(walkId numWalker, PartitionStrategy &partitions, App *app, int gpuId):
        _partitions(partitions),
        _numPartition(partitions.numPartition()),
        _devicePartitions(partitions.devicePtr()),
        _numPage(numPageRequired(numWalker)),
        _maxWalker(numWalker),
        _rand(numBlock, threadPerBlock, gpuId),
        _app(app),
        _active(_numPartition, pageSize, numWalker, gpuId),
        _gpuId(gpuId)
    {}

    walkId numPage() const {
        return _numPage;
    }

    void insert(GPUInitWalker<Walker> &initWalker, WalkManager<Walker> &walkman, CUDAStream &stream) {
        GPUVectorPool<Walker> walkers(1, pageSize, _gpuId);
        initWalker.reset();

        while (!initWalker.done()) {
            walkId length = pageSize;
            initWalker.create(walkers[0], length, _rand, stream);
            insertPushedToQueue<<<numBlock, threadPerBlock, 0, stream.get()>>>(_active.devicePtr(), walkers[0], length, _numPartition, _devicePartitions);
            _active.insertFetchedPage(stream);

            if (initWalker.generated() > _maxWalker) {
                _active.copyLengthMetaData(stream);
                stream.sync();

                partitionId p = getOutMemPartition(_partitions.numPartition());
                walkman.setEvictedWalker(_active, p, stream);
                stream.sync();
            }
        }
    }

    void processInMemory(partitionId partition, const edgeId *rowptr, const nodeId *col, CUDAStream &stream) 
    {
        const nodeId nodeBegin = _partitions.nodeOffset(partition);
        const nodeId nodeEnd = _partitions.nodeOffset(partition + 1);
        const edgeId edgeBegin = _partitions.edgeOffset(partition);

        randomWalkInsidePool<<<numBlock, threadPerBlock, 0, stream.get()>>> \
        (partition, nodeBegin, nodeEnd, edgeBegin, rowptr, col, _rand.getRandState(), _active.devicePtr(), _devicePartitions, _app);

        insertReservedToQueue<<<numBlock, threadPerBlock, 0, stream.get()>>>(_active.devicePtr(), partition, _numPartition, _devicePartitions);
        _active.freeFirstPageAndInsertFetchedPage(partition, stream);
    }

    void processInMemory(partitionId partition, const edgeId *rowptr, const nodeId *col, CUDAStream &stream, int times)
    {
        const nodeId nodeBegin = _partitions.nodeOffset(partition);
        const nodeId nodeEnd = _partitions.nodeOffset(partition + 1);
        const edgeId edgeBegin = _partitions.edgeOffset(partition);

        randomWalkInsidePoolAllBatch<<<numBlock, threadPerBlock, 0, stream.get()>>> \
        (times, partition, nodeBegin, nodeEnd, edgeBegin, rowptr, col, _rand.getRandState(), _active.devicePtr(), _devicePartitions, _app);

        for (int i = 0; i < times; i++) {
            insertReservedToQueue<<<numBlock, threadPerBlock, 0, stream.get()>>>(_active.devicePtr(), partition, _numPartition, _devicePartitions);
            _active.freeFirstPageAndInsertFetchedPage(partition, stream);
        }
    }
    
    void processOffMemory(GPUBuffer<Walker> &batch, partitionId partition, const edgeId *rowptr, const nodeId *col, CUDAStream &stream) 
    {
        const nodeId nodeBegin = _partitions.nodeOffset(partition);
        const nodeId nodeEnd = _partitions.nodeOffset(partition + 1);
        const edgeId edgeBegin = _partitions.edgeOffset(partition);

        randomWalkOutsidePool<<<numBlock, threadPerBlock, 0, stream.get()>>>  \
        (batch.ptr(), batch.length(), nodeBegin, nodeEnd, edgeBegin, rowptr, col, _rand.getRandState(), _active.devicePtr(), _devicePartitions, _app);

        
        insertPushedToQueue<<<numBlock, threadPerBlock, 0, stream.get()>>>(_active.devicePtr(), batch.ptr(), batch.length(), _numPartition, _devicePartitions);
        _active.pushFreePage(stream, batch);

        _active.insertFetchedPage(stream);
    }

    walkId length(partitionId i) {
        return _active.totalLength(i);
    }

    walkId numWalker() {
        walkId sum = 0;
        for (partitionId i = 0; i < _numPartition; i++) {
            sum += length(i);
        }
        return sum;
    }

    walkId numBatch(partitionId i) {
        return _active.totalBatch(i);
    }
};
