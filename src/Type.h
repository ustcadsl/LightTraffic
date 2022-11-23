#pragma once

#include <cub/cub.cuh>

using nodeId = u_int32_t;
using edgeId = u_int64_t;
using walkId = u_int64_t;
using partitionId = u_int16_t;

const int numBlock = 82;
const int threadPerBlock = 512;
const int sharedMemPerBlock = 49152;
const int constantMem = 48 * 1024;
const int walkerPerThread = 4;
const int pageSize = numBlock * threadPerBlock * walkerPerThread;
const int cachelineSize = 128;

const nodeId INVALID_NODE = 0xFFFFFFFF;
const u_int32_t INVALID_WALK = 0xFFFFFFFF;
const partitionId INVALID_PART = 0xFFFF;

using BlockScan = cub::BlockScan<u_int32_t, threadPerBlock>;
