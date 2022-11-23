#pragma once

#include <vector>
#include <unordered_map>
#include <utility>
#include <memory>

#include "Type.h"
#include "WalkManager.h"
#include "partition/PartitionStrategy.h"
#include "buffer/Pool.h"

class DiskGraph
{
private:
    std::vector<std::unique_ptr<DiskBuffer<edgeId>>> _rowptrs;
    std::vector<std::unique_ptr<DiskBuffer<nodeId>>> _cols;

    std::vector<std::unique_ptr<DiskBuffer<edgeId>>> _rowptr;
    std::vector<std::unique_ptr<DiskBuffer<nodeId>>> _col;

    FILE *_fp;

public:

    DiskGraph(FILE *fp, PartitionStrategy &partitionStrategy): DiskGraph(fp) {
        setPartition(partitionStrategy);
    }

    DiskGraph(FILE *fp): _fp(fp) {
        nodeId numNode;
        edgeId numEdge;

        fseek(fp, 0, SEEK_SET);
        fread(&numNode, sizeof(nodeId), 1, fp);
        fread(&numEdge, sizeof(edgeId), 1, fp);

        size_t rowptrOffset = sizeof(nodeId) + sizeof(edgeId);
        size_t colOffset = rowptrOffset + sizeof(edgeId) * (numNode + 1);

        _rowptr.emplace_back(std::make_unique<DiskBuffer<edgeId>>(
            fp, rowptrOffset, numNode + 1
        ));

        _col.emplace_back(std::make_unique<DiskBuffer<nodeId>>(
            fp, colOffset, numEdge
        ));
    }

    void setPartition(PartitionStrategy &partitionStrategy) {
        partitionId numPartition = partitionStrategy.numPartition();
        size_t rowptrOffset = sizeof(nodeId) + sizeof(edgeId);
        size_t colOffset = rowptrOffset + sizeof(edgeId) * (partitionStrategy.numNode() + 1);

        for (partitionId i = 0; i < numPartition; i++) {
            _rowptrs.emplace_back(std::make_unique<DiskBuffer<edgeId>>(
                _fp, rowptrOffset + sizeof(edgeId) * partitionStrategy.nodeOffset(i), partitionStrategy.nodeOffset(i + 1) - partitionStrategy.nodeOffset(i) + 1
            ));

            _cols.emplace_back(std::make_unique<DiskBuffer<nodeId>>(
                _fp, colOffset + sizeof(nodeId) * partitionStrategy.edgeOffset(i), partitionStrategy.edgeOffset(i + 1) - partitionStrategy.edgeOffset(i)
            ));
        }
    }

    auto& rowptr(partitionId p)
    {
        assert(p < _rowptrs.size());
        return _rowptrs[p];
    }

    auto& col(partitionId p)
    {
        assert(p < _cols.size());
        return _cols[p];
    }
    
    auto& rowptr()
    {
        return _rowptr[0];
    }

    auto& col()
    {
        return _col[0];
    }
};

class CPUGraph
{
private:
    std::vector<std::unique_ptr<CPUBuffer<edgeId>>> _rowptrs;
    std::vector<std::unique_ptr<CPUBuffer<nodeId>>> _cols;

    CPUBuffer<edgeId> _rowptr;
    CPUBuffer<nodeId> _col;

public:
    CPUGraph(partitionId numInMemoryPartition, PartitionStrategy &partitionStrategy):
        _rowptr(partitionStrategy.numNode() + 1),
        _col(partitionStrategy.numEdge())
    {
        setPartition(partitionStrategy);
    }

    void setPartition(PartitionStrategy &partitionStrategy) {
        partitionId numPartition = partitionStrategy.numPartition();
        _rowptrs.clear();
        _cols.clear();

        for (partitionId i = 0; i < numPartition; i++) {
            _rowptrs.emplace_back(std::make_unique<CPUBuffer<edgeId>>(
                partitionStrategy.nodeOffset(i + 1) - partitionStrategy.nodeOffset(i) + 1, _rowptr.ptr() + partitionStrategy.nodeOffset(i)
            ));

            _cols.emplace_back(std::make_unique<CPUBuffer<nodeId>>(
                partitionStrategy.edgeOffset(i + 1) - partitionStrategy.edgeOffset(i), _col.ptr() + partitionStrategy.edgeOffset(i)
            ));
        }
    }

    void load(DiskGraph &diskGraph) {
        _rowptr.from(*diskGraph.rowptr());
        _col.from(*diskGraph.col());
    }

    auto& rowptr(partitionId p)
    {
        assert(p < _rowptrs.size());
        return _rowptrs[p];
    }

    auto& col(partitionId p)
    {
        assert(p < _cols.size());
        return _cols[p];
    }

    auto& rowptr()
    {
        return _rowptr;
    }

    auto& col()
    {
        return _col;
    }

    bool existed(partitionId p)
    {
        return true;
    }

    void cudaRegister() {
        _rowptr.cudaRegister();
        _col.cudaRegister();
    }    
};

class GPUGraph
{
private:
    GPUVectorPool<u_int8_t> _graph;

    std::unordered_map<partitionId, size_t> _index;
    std::vector<size_t> _free;

    std::unordered_map<partitionId, size_t> _col_offset;

    partitionId _numInMemoryPartition;

public:
    int loadedGraphNum = 0;

    GPUGraph(partitionId numInMemoryPartition, PartitionStrategy &partitionStrategy, int gpuId):
        _graph(numInMemoryPartition, partitionStrategy.partitionSize(), gpuId),
        _numInMemoryPartition(numInMemoryPartition)
    {        
        for (partitionId i = 0; i < _numInMemoryPartition; i++) {
            _free.emplace_back(i);
        }
    }    

    void evict(partitionId p)
    {
        assert(existed(p));
        size_t i = _index[p];
        _index.erase(p);

        _free.push_back(i);
    }

    void loadAsync(partitionId p, CPUGraph &cpuGraph, CUDAStream &stream) {
        if (!existed(p)) {
            if (_free.size() == 0) {
                evict(_index.begin()->first);
            }

            size_t i = _free.back();
            _free.pop_back();

            CPUBuffer<u_int8_t> rowptr(*cpuGraph.rowptr(p));
            CPUBuffer<u_int8_t> col(*cpuGraph.col(p));

            _col_offset[p] = rowptr.sizeInByte();

            _graph.readAsync(i, rowptr, stream);
            _graph.readAsync(i, col, _col_offset[p], stream);

            _index[p] = i;

            loadedGraphNum++;
        }
    }

    auto rowptr(partitionId p)
    {
        assert(existed(p));
        return (edgeId *)_graph[_index[p]];
    }

    auto col(partitionId p)
    {
        assert(existed(p));
        return (nodeId *)(_graph[_index[p]] + _col_offset[p]);
    }

    bool existed(partitionId p)
    {
        return _index.count(p) > 0;
    }

    bool hasFreeSpace() {
        return _free.size() > 0;
    }
    
    auto existPartitions() {
        return std::pair<decltype(_index.cbegin()), decltype(_index.cbegin())>(_index.cbegin(), _index.cend());
    }
};
