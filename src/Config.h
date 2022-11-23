#pragma once

#include <sstream>

#include "anyoption.h"
#include "Type.h"
#include "partition/PartitionStrategy.h"

class Config {
private:
    template <typename T>
    T getNum(const char *str, T defaultValue) {
        if (str != NULL) {
            T num;
            std::istringstream iss(str);
            iss >> num;
            return num;
        }
        else {
            return defaultValue;
        }
    }

public:
    FILE *_file;
    PartitionStrategy *_partitions;
    partitionId _numDevicePartition;
    walkId _numWalker;
    walkId _numDeviceWalker;
    size_t _partition_size;

    nodeId _step;
    int _runs;
    nodeId _trimmed;
    nodeId _source;
    float _prob;

    Config(int argc, char* argv[]) {
        AnyOption *opt = new AnyOption();
        opt->setOption("file", 'f');
        opt->setOption("partition", 'p');
        opt->setOption("batch", 'b');
        opt->setOption("walker", 'w');
        opt->setOption("partition-size", 's');
        
        opt->setOption("length", 'l');
        opt->setOption("runs");
        opt->setOption("trimmed");
        opt->setOption("source");
        opt->setOption("prob");

        opt->setFlag("materialized-partition", 'm');
        

        opt->processCommandArgs(argc, argv);

        assert(opt->getValue("file") != NULL);
        char *filename = opt->getValue("file");
        _file = fopen(filename, "r");
        assert(_file != NULL);

        if (opt->getFlag('m')) {
            _partition_size = getNum<size_t>(opt->getValue('s'), 32);
            _partitions = new EqualSizeStrategy(_file, 1024 * 1024 * _partition_size);
            _partitions->materialize(filename);
        }
        else {
            _partitions = new MaterializedStrategy(filename);
        }

        _numDevicePartition = std::min(_partitions->numPartition(), getNum<partitionId>(opt->getValue('p'), _partitions->numPartition()));
        _numWalker = getNum<walkId>(opt->getValue('w'), _partitions->numNode() * 2);
        _numDeviceWalker = std::min(_numWalker, getNum<walkId>(opt->getValue('b'), _numWalker));
        if (_numDeviceWalker != _numWalker && _numDeviceWalker <= _partitions->numPartition() * pageSize) {
            printf("GPU memory pool for walkers should be capable to save at least %u walkers for memory safety!\n", _partitions->numPartition() * pageSize);
        }
        if (sizeof(BlockScan::TempStorage) + sizeof(u_int32_t) * (_partitions->numPartition() * 2 + 1 + 2 * walkerPerThread * threadPerBlock) + sizeof(partitionId) * walkerPerThread * threadPerBlock > sharedMemPerBlock) {
            printf("GPU shared memory overflow! Please consider reduce the number of partitions or batch size.\n");
            exit(1);
        }

        _step = getNum<nodeId>(opt->getValue('l'), 10);
        _runs = getNum<int>(opt->getValue("runs"), 1);
        _trimmed = getNum<nodeId>(opt->getValue("trimmed"), _step);
        _source = getNum<nodeId>(opt->getValue("source"), 0);
        _prob = getNum<float>(opt->getValue("prob"), 0.15);
        
        printf("num nodes: %u, num edges: %lu, num walkers: %lu\n", _partitions->numNode(), _partitions->numEdge(), _numWalker);
    }

    ~Config() {
        fclose(_file);
        delete _partitions;
    }
};
