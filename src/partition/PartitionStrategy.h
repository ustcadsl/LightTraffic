#pragma once

#include <unistd.h>
#include <assert.h>

#include <vector>

#include "Type.h"
#include "buffer/Buffer.h"

#include "utils/GPUCall.h"

__constant__ nodeId partitionNodeOffset[constantMem / sizeof(nodeId)];

class DevicePartition {
public:
    nodeId _maxNumNodeInPartition;
    partitionId _numPartition;

    __device__ partitionId findPartition(nodeId n) const {       
        partitionId low = 0;
        partitionId high = _numPartition;

        while (low < high) {
            partitionId mid = low + (high - low + 1) / 2;
            if (n < partitionNodeOffset[mid]) {
                high = mid - 1;
            }
            else {
                low = mid;
            }
        }

        return low;
        
    }

    __device__ partitionId numPartition() const
    {
        return _numPartition;
    }
};

void copyPartitionNodeOffset(nodeId* nodeOffset, partitionId numPartition) {
    gpuCall(cudaMemcpyToSymbol(partitionNodeOffset, nodeOffset, sizeof(nodeId) * (numPartition + 1)));
}

class PartitionStrategy {
public:
    virtual nodeId nodeOffset(partitionId i) const = 0;
    virtual edgeId edgeOffset(partitionId i) const = 0;
    virtual nodeId numNode() const = 0;
    virtual edgeId numEdge() const = 0;
    virtual partitionId numPartition() const = 0;
    virtual nodeId maxNumNodeInPartition() const = 0;
    virtual edgeId maxNumEdgeInPartition() const = 0;
    virtual size_t partitionSize() const = 0;
    virtual partitionId findPartition(nodeId n) const = 0;
    virtual void materialize(std::string filename) = 0;

    virtual DevicePartition* devicePtr() = 0;
    virtual ~PartitionStrategy(){}
};

class MaterializedStrategy: public PartitionStrategy {
private:
    partitionId _numPartition;
    size_t _partition_size;
    nodeId *_nodeOffset;
    edgeId *_edgeOffset;

    nodeId _maxNumNodeInPartition{0};
    edgeId _maxNumEdgeInPartition{0};

public:
    MaterializedStrategy(std::string filename) {
        std::string partitionFileName = filename + ".part"; 

        FILE *fp = fopen(partitionFileName.c_str(), "r");
        assert(fp != NULL);

        fread(&_numPartition, sizeof(partitionId), 1, fp);
        fread(&_partition_size, sizeof(size_t), 1, fp);

        _nodeOffset = new nodeId[_numPartition + 1];
        _edgeOffset = new edgeId[_numPartition + 1];

        fread(_nodeOffset, sizeof(nodeId), _numPartition + 1, fp);
        fread(_edgeOffset, sizeof(edgeId), _numPartition + 1, fp);

        for (partitionId i = 0; i < numPartition(); i++) {
            _maxNumNodeInPartition = std::max(_maxNumNodeInPartition, nodeOffset(i + 1) - nodeOffset(i));
            _maxNumEdgeInPartition = std::max(_maxNumEdgeInPartition, edgeOffset(i + 1) - edgeOffset(i));
        }

        fclose(fp);
    }

    DevicePartition* devicePtr() {
        DevicePartition *_device;
        DevicePartition host{_maxNumNodeInPartition, _numPartition};

        copyPartitionNodeOffset(_nodeOffset, numPartition());

        gpuCall(cudaMalloc(&_device, sizeof(DevicePartition)));
        gpuCall(cudaMemcpy(_device, &host, sizeof(DevicePartition), cudaMemcpyHostToDevice));

        return _device;
    }

    nodeId nodeOffset(partitionId i) const override
    {
        return _nodeOffset[i];
    }

    edgeId edgeOffset(partitionId i) const override
    {
        return _edgeOffset[i];
    }

    nodeId numNode() const override
    {
        return _nodeOffset[_numPartition];
    }

    edgeId numEdge() const override
    {
        return _edgeOffset[_numPartition];
    }

    partitionId numPartition() const override
    {
        return _numPartition;
    }

    nodeId maxNumNodeInPartition() const override
    {
        return _maxNumNodeInPartition;
    }

    edgeId maxNumEdgeInPartition() const override
    {
        return _maxNumEdgeInPartition;
    }

    size_t partitionSize() const override
    {
        return _partition_size;
    }

    partitionId findPartition(nodeId n) const override
    {
        assert(n < numNode());
        partitionId low = 0;
        partitionId high = numPartition() - 1;

        while (low < high) {
            partitionId mid = low + (high - low + 1) / 2;
            if (n < nodeOffset(mid)) {
                high = mid - 1;
            }
            else {
                low = mid;
            }
        }

        return low;
    }

    void materialize(std::string filename) override {}

    ~MaterializedStrategy()
    {
        delete[] _nodeOffset;
        delete[] _edgeOffset;
    }
};

class EqualSizeStrategy: public PartitionStrategy {
private:
    std::vector<nodeId> _nodeOffset;
    std::vector<edgeId> _edgeOffset;

    nodeId _maxNumNodeInPartition{0};
    edgeId _maxNumEdgeInPartition{0};

    size_t _partition_size;

    size_t partitionSize(edgeId *rowptr, nodeId begin, nodeId end) {
        return (end - begin + 1) * sizeof(edgeId) + (rowptr[end] - rowptr[begin]) * sizeof(nodeId);
    }

    void init(CPUBuffer<edgeId> &rowptr) {
        nodeId n = 0;
        _nodeOffset.emplace_back(0);
        _edgeOffset.emplace_back(0);
        nodeId numNode = rowptr.length() - 1;
        
        for (nodeId i = 1; i <= numNode;) {
            if (partitionSize(rowptr.ptr(), n, i) > _partition_size) {
                if (i == n + 1) {
                    std::stringstream ss;
                    ss << "Node " << n << " has degree " << rowptr[i] - rowptr[n] << ". It cannot fit into partition size " << _partition_size; 
                    throw std::runtime_error(ss.str());
                }
                else {
                    n = i - 1;
                    _nodeOffset.emplace_back(n);
                    _edgeOffset.emplace_back(rowptr[n]);
                }
            }
            else {
                i++;
            }
        }

        _nodeOffset.emplace_back(numNode);
        _edgeOffset.emplace_back(rowptr[numNode]);

        for (partitionId i = 0; i < numPartition(); i++) {
            _maxNumNodeInPartition = std::max(_maxNumNodeInPartition, nodeOffset(i + 1) - nodeOffset(i));
            _maxNumEdgeInPartition = std::max(_maxNumEdgeInPartition, edgeOffset(i + 1) - edgeOffset(i));
        }

        edgeId maxDegree = 0;
        nodeId nodeWithMaxDegree;
        for (nodeId i = 1; i < rowptr.length(); i++) {
            if (maxDegree < rowptr[i] - rowptr[i - 1]) {
                maxDegree = rowptr[i] - rowptr[i - 1];
                nodeWithMaxDegree = i - 1;
            }
        }

        printf("node with max degree: %u, degree: %lu\n", nodeWithMaxDegree, maxDegree);
    }

public:

    EqualSizeStrategy(CPUBuffer<edgeId> &rowptr, size_t maxSize): _partition_size(maxSize) {
        init(rowptr);
    }

    EqualSizeStrategy(FILE *fp, size_t maxSize): _partition_size(maxSize) {
        nodeId numNode;

        pread64(fileno(fp), &numNode, sizeof(nodeId), 0);

        CPUBuffer<edgeId> rowptr(numNode + 1);
        DiskBuffer<edgeId> disk(fp, sizeof(nodeId) + sizeof(edgeId), numNode + 1);

        disk.to(rowptr);
        init(rowptr);
    }

    DevicePartition* devicePtr() {
        DevicePartition *_device;
        DevicePartition host{_maxNumNodeInPartition, numPartition()};
        copyPartitionNodeOffset(_nodeOffset.data(), numPartition());

        gpuCall(cudaMalloc(&_device, sizeof(DevicePartition)));
        gpuCall(cudaMemcpy(_device, &host, sizeof(DevicePartition), cudaMemcpyHostToDevice));

        return _device;
    }

    nodeId nodeOffset(partitionId i) const override
    {
        return _nodeOffset[i];
    }

    edgeId edgeOffset(partitionId i) const override
    {
        return _edgeOffset[i];
    }

    nodeId numNode() const override
    {
        return _nodeOffset[numPartition()];
    }

    edgeId numEdge() const override
    {
        return _edgeOffset[numPartition()];
    }

    partitionId numPartition() const override
    {
        return _nodeOffset.size() - 1;
    }

    nodeId maxNumNodeInPartition() const override
    {
        return _maxNumNodeInPartition;
    }

    edgeId maxNumEdgeInPartition() const override
    {
        return _maxNumEdgeInPartition;
    }

    size_t partitionSize() const override
    {
        return _partition_size;
    }

    partitionId findPartition(nodeId n) const override
    {
        assert(n < numNode());
        partitionId low = 0;
        partitionId high = numPartition() - 1;

        while (low < high) {
            partitionId mid = low + (high - low + 1) / 2;
            if (n < nodeOffset(mid)) {
                high = mid - 1;
            }
            else {
                low = mid;
            }
        }

        return low;
    }

    void materialize(std::string filename) {
        std::string partitionFileName = filename + ".part"; 

        FILE *fp = fopen(partitionFileName.c_str(), "w");
        partitionId numPart = numPartition();

        fwrite(&numPart, sizeof(partitionId), 1, fp);
        fwrite(&_partition_size, sizeof(size_t), 1, fp);
        fwrite(_nodeOffset.data(), sizeof(nodeId), numPart + 1, fp);
        fwrite(_edgeOffset.data(), sizeof(edgeId), numPart + 1, fp);

        fclose(fp);
    }
};
