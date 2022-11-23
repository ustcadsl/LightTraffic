#pragma once

#include <vector>
#include <list>
#include <memory>

#include "buffer/Buffer.h"
#include "utils/Event.h"

template <typename T>
class VectorPool {
private:
    T **_pool;
    size_t _numPage;
    size_t _pageCapacity;

    T* _ptr;
    size_t *_length;

    bool cudaPinned{false};

public:
    VectorPool(const VectorPool<T>&) = delete;
    VectorPool(VectorPool<T>&&) = delete;

    VectorPool& operator=(const VectorPool<T>&) = delete;
    VectorPool& operator=(VectorPool<T>&&) = delete;

    VectorPool(size_t numPage, size_t pageCapacity):
        _numPage(numPage), _pageCapacity(pageCapacity)
    {
        _ptr = new T[numPage * pageCapacity];
        _pool = new T*[numPage];
        _length = new size_t[numPage];

        for (size_t i = 0; i < numPage; i++) {
            _pool[i] = _ptr + pageCapacity * i;
            _length[i] = 0;
        }
    }

    ~VectorPool() {
        if (cudaPinned)
            cudaHostUnregister(_ptr);

        delete[] _ptr;
        delete[] _pool;
        delete[] _length;
    }

    T* operator[](size_t i) {
        return _pool[i];
    }

    void read(size_t i, DiskBuffer<T> &disk) {
        CPUBuffer<T> buf(_pageCapacity, _pool[i]);
        disk.to(buf);
        _length[i] = buf.length();
    }

    void append(size_t i, T &element) {
        assert(_length[i] < _pageCapacity);
        _pool[_length[i]] = element;
        _length[i] += 1;
    }

    void clear(size_t i) {
        _length[i] = 0;
    }

    size_t length(size_t i) {
        return _length[i];
    }

    T* begin(size_t i) {
        return _pool[i];
    }

    T* end(size_t i) {
        return _pool[i] + _length[i];
    }

    void cudaRegister() {
        if (!cudaPinned)
            cudaHostRegister((void *)_ptr, _numPage * _pageCapacity * sizeof(T), 0);
        cudaPinned = true;
    }
};

template <typename T>
class ListPool {
private:

    u_int32_t _numBuffer;
    u_int32_t _listCapacity;
    size_t _pageCapacity;

    u_int32_t _numPage;

    CPUBuffer<T> _pool;

    CPUBuffer<u_int32_t> _frame;
    CPUBuffer<u_int32_t> _length;

    CPUBuffer<u_int32_t> _head;
    CPUBuffer<u_int32_t> _rear;

    CPUBuffer<u_int32_t> _free;

    u_int32_t _freeHead;
    u_int32_t _freeRear;

    u_int32_t getPageId(u_int32_t i, u_int32_t j) {
        return _frame[i * _listCapacity + j];
    }

    T* getPageAddr(u_int32_t i, u_int32_t j) {
        return _pool.ptr() + getPageId(i, j) * _pageCapacity;
    }

    u_int32_t getIndex(u_int32_t* ptr) {
        u_int32_t index = *ptr;
        *ptr = (*ptr + 1) % _listCapacity;

        return index;
    }

    void fetchPage(u_int32_t i) {
        getIndex(&_rear[i]);
        _frame[i * _listCapacity + _rear[i]] = _free[getIndex(&_freeHead)];
    }

    void freePage(u_int32_t i) {
        getIndex(&_freeRear);
        _free[_freeRear] = _frame[i * _listCapacity + getIndex(&_head[i])];
    }

public:
    ListPool(const ListPool<T>&) = delete;
    ListPool(ListPool<T>&&) = delete;

    ListPool& operator=(const ListPool<T>&) = delete;
    ListPool& operator=(ListPool<T>&&) = delete;

    ListPool(u_int32_t numBuffer, size_t pageCapacity, size_t numElement):
        _numBuffer(numBuffer), _listCapacity((numElement + pageCapacity - 1) / pageCapacity + 2), _pageCapacity(pageCapacity), _numPage(_listCapacity + numBuffer),\
        _pool(_numPage * pageCapacity), _frame(numBuffer * _listCapacity), _length(numBuffer), _head(numBuffer), _rear(numBuffer), _free(_listCapacity),\
        _freeHead(0), _freeRear(_listCapacity - 1)
    {
        for (u_int32_t i = 0; i < numBuffer; i++) {
            _frame[i * _listCapacity] = i + _listCapacity;

            _head[i] = 0;
            _rear[i] = 0;
            _length[i] = 0;
        }

        for (u_int32_t i = 0; i < _listCapacity; i++) {
            _free[i] = i;
        }
    }

    void append(u_int32_t i, const T &element) {        
        if (_length[i] == _pageCapacity) {
            fetchPage(i);
            _length[i] = 0;
        }

        getPageAddr(i, _rear[i])[_length[i]] = element;
        _length[i] += 1;
    }

    void clear(u_int32_t i) {
        if (_head[i] == _rear[i]) {
            _length[i] = 0;
            return;
        }

        freePage(i);
    }

    auto getFreePage() {
        assert(_freeHead != _freeRear);
        return CPUBuffer<T>(0, _pool.ptr() + _free[getIndex(&_freeHead)] * _pageCapacity);
    }

    u_int32_t totalBatch(u_int32_t i) {
        return (_rear[i] + _pageCapacity - _head[i]) % _pageCapacity + 1;
    }

    size_t length(u_int32_t i) {
        return (totalBatch(i) - 1) * _pageCapacity + _length[i];
    }
    
    T* begin(u_int32_t i) {
        return getPageAddr(i, _head[i]);
    }

    T* end(u_int32_t i) {
        return getPageAddr(i, _head[i]) + (totalBatch(i) == 1? _length[i]: _pageCapacity);
    }
    
    auto removePage(u_int32_t i) {
        CPUBuffer<T> buffer(end(i) - begin(i), begin(i));
        
        if (_head[i] == _rear[i]) {
            _length[i] = 0;
            _frame[i * _listCapacity + _head[i]] = _free[getIndex(&_freeHead)];
        }
        else {
            getIndex(&_head[i]);
        }

        return buffer;
    }

    void insertFreePage(CPUBuffer<T> &buffer) {
        getIndex(&_freeRear);
        _free[_freeRear] = (buffer.ptr() - _pool.ptr()) / _pageCapacity;
    }

    void insertPage(u_int32_t i, CPUBuffer<T> &buffer) {

        u_int32_t prevPageId = getPageId(i, _rear[i]);
        _frame[i * _listCapacity + _rear[i]] = (buffer.ptr() - _pool.ptr()) / _pageCapacity;

        if (buffer.length() == _pageCapacity) {            
            if (length(i) == 0) {
                getIndex(&_freeRear);
                _free[_freeRear] = prevPageId;
                _length[i] = _pageCapacity;
            }
            else {
                getIndex(&_rear[i]);
                _frame[i * _listCapacity + _rear[i]] = prevPageId;
            }
            return;
        }

        u_int32_t copySize = std::min((u_int32_t)(_pageCapacity - buffer.length()), _length[i]);
        for (int j = 0; j < copySize; j++) {
            buffer[buffer.length() + j] = _pool[prevPageId * _pageCapacity + _length[i] - copySize + j];
        }

        if (_length[i] == copySize) {
            getIndex(&_freeRear);
            _free[_freeRear] = prevPageId;
            _length[i] += buffer.length();
        }
        else {
            getIndex(&_rear[i]);
            _frame[i * _listCapacity + _rear[i]] = prevPageId;
            _length[i] -= copySize;
        }
    }

    void cudaRegister() {
        _pool.cudaRegister();
    }
};

template <typename T>
class DeviceVectorPool 
{
public:
    T **_pool;
    size_t *_length;

    __device__ void append(size_t i, T &element) {
        _pool[i][_length[i]] = element;
        _length[i] += 1;
    }

    __device__ void clear(size_t i) {
        _length[i] = 0;
    }

    __device__ size_t length(size_t i) {
        return _length[i];
    }

    __device__ T* begin(size_t i) {
        return _pool[i];
    }

    __device__ T* end(size_t i) {
        return _pool[i] + _length[i];
    }
};

template <typename T>
static __global__ void _initDeviceVectorPool(DeviceVectorPool<T>* p, T* ptr, size_t numPage, size_t pageCapacity) {

    const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numPage) {
        p->_pool[tid] = ptr + pageCapacity * tid;
    }
}

template <typename T>
class GPUVectorPool {
private:
    GPUBuffer<T*> _pool;
    size_t _numBuffer;
    size_t _pageCapacity;

    GPUBuffer<T> _ptr;
    GPUBuffer<size_t> _length;

    DeviceVectorPool<T> *_device;

    int _gpuId;

public:
    GPUVectorPool(const GPUVectorPool<T>&) = delete;
    GPUVectorPool(GPUVectorPool<T>&&) = delete;

    GPUVectorPool& operator=(const GPUVectorPool<T>&) = delete;
    GPUVectorPool& operator=(GPUVectorPool<T>&&) = delete;

    GPUVectorPool(size_t numBuffer, size_t pageCapacity, int gpuId):
        _pool(numBuffer, gpuId), _numBuffer(numBuffer), _pageCapacity(pageCapacity), _ptr(numBuffer * pageCapacity, gpuId), _length(numBuffer, gpuId), _gpuId(gpuId)
    {
        gpuCall(cudaMalloc(&_device, sizeof(DeviceVectorPool<T>)));

        DeviceVectorPool<T> host{_pool.ptr(), _length.ptr()};
        gpuCall(cudaMemcpy(_device, &host, sizeof(DeviceVectorPool<T>), cudaMemcpyHostToDevice));

        _initDeviceVectorPool<T><<<numBuffer / threadPerBlock + 1, threadPerBlock>>>(_device, _ptr.ptr(), numBuffer, pageCapacity);
    }

    ~GPUVectorPool() {
        gpuCall(cudaFree(_device));
    }

    DeviceVectorPool<T>* devicePtr() {
        return _device;
    }

    T* operator[](size_t i) {
        return _ptr.ptr() + i * _pageCapacity;
    }
    
    void read(size_t i, CPUBuffer<T> &buffer) {
        size_t length = buffer.length();

        GPUBuffer<T> device_buffer(length, _ptr.ptr() + i * _pageCapacity, _gpuId);
        buffer.to(device_buffer);
    }

    void readAsync(size_t i, CPUBuffer<T> &buffer, CUDAStream &stream) {
        size_t length = buffer.length();

        GPUBuffer<T> device_buffer(length, _ptr.ptr() + i * _pageCapacity, _gpuId);
        buffer.toAsync(device_buffer, stream);
    }

    void readAsync(size_t i, CPUBuffer<T> &buffer, size_t offset, CUDAStream &stream) {
        if (offset % sizeof(T) != 0) {
            offset += sizeof(T) - (offset % sizeof(T));
        }

        size_t length = buffer.length();

        GPUBuffer<T> device_buffer(length, _ptr.ptr() + i * _pageCapacity + offset, _gpuId);
        buffer.toAsync(device_buffer, stream);
    }
};


template <typename T>
class DeviceListPool {
public:
    T *_pool;

    u_int32_t *_frame;

    u_int32_t *_length;

    u_int32_t *_head;
    u_int32_t *_rear;

    T **_rearPage;

    u_int32_t *_free;

    u_int32_t _freeHead;
    u_int32_t _freeRear;

    size_t _pageCapacity;
    u_int32_t _listCapacity;    

    u_int32_t _numBuffer;

    __device__ u_int32_t getPageId(u_int32_t i, u_int32_t j) {
        return _frame[i * _listCapacity + j];
    }

    __device__ T* getPageAddr(u_int32_t i, u_int32_t j) {
        return _pool + getPageId(i, j) * _pageCapacity;
    }

    __device__ u_int32_t getIndex(u_int32_t* ptr) {
        u_int32_t index = atomicAdd(ptr, 1U);
        if (index == _listCapacity - 1) {
            atomicSub(ptr, _listCapacity);
        }

        return index % _listCapacity;
    }

    __device__ u_int32_t headPageLength(u_int32_t i) {
        return _head[i] == _rear[i]? _length[i]: _pageCapacity;
    }

    __device__ void fetchPage(u_int32_t i) {
        assert(_freeHead != _freeRear);

        u_int32_t pageId = _free[getIndex(&_freeHead)];
        u_int32_t oldRear = getIndex(&_rear[i]);

        u_int32_t frameId = i * _listCapacity + (oldRear + 2) % _listCapacity;
        _frame[frameId] = pageId;
    }

    __device__ void freePage(u_int32_t i) {
        u_int32_t pageId = getPageId(i, getIndex(&_head[i]));
        u_int32_t oldRear = getIndex(&_freeRear);
        
        _free[(oldRear + 1) % _listCapacity] = pageId;
    }

    __device__ void freeFirstPage(u_int32_t i) {
        if (_head[i] == _rear[i]) {
            fetchPage(i);
            _length[i] = 0;
        }

        freePage(i);
    }

    __device__ T* getHead(u_int32_t i) {
        return getPageAddr(i, _head[i]);
    }

    __device__ T* getRear(u_int32_t i, u_int32_t j) {
        assert(j <= 1);
        return getPageAddr(i, (_rear[i] + j) % _listCapacity);
    }

    __device__ void handleOverflow() {
        u_int32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        for (u_int32_t i = tid; i < _numBuffer; i += gridDim.x * blockDim.x) {
            if (_length[i] > _pageCapacity) {
                fetchPage(i);
                _length[i] -= _pageCapacity;
            }
        }
    }
};

template <typename T>
static __global__ void _initDeviceListPool(DeviceListPool<T>* p, u_int32_t numBuffer, u_int32_t listCapacity) {
    const u_int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numBuffer) {
        p->_frame[tid * listCapacity] = tid * 2 + listCapacity;
        p->_frame[tid * listCapacity + 1] = tid * 2 + 1 + listCapacity;

        p->_rearPage[tid * 2] = p->_pool + (tid * 2 + listCapacity) * p->_pageCapacity;
        p->_rearPage[tid * 2 + 1] = p->_pool + (tid * 2 + 1 + listCapacity) * p->_pageCapacity;

        p->_head[tid] = 0;
        p->_rear[tid] = 0;
        p->_length[tid] = 0;
    }
    else if (tid < numBuffer + listCapacity) {
        p->_free[tid - numBuffer] = tid - numBuffer;
    }
}

template <typename T>
static __global__ void page_freeFirstAndInsertFetched(DeviceListPool<T>* p, u_int32_t i) {
    if (threadIdx.x == 0) {
        p->freeFirstPage(i);
    }
    __syncthreads();
    
    p->handleOverflow();
}

template <typename T>
static __global__ void page_insertFetched(DeviceListPool<T>* p) {
    p->handleOverflow();
}

template <typename T>
static __global__ void page_popFirst(DeviceListPool<T>* p, u_int32_t i, T** pagePtr) {
    if (p->_head[i] == p->_rear[i]) {
        p->fetchPage(i);

        pagePtr[0] = p->getPageAddr(i, p->getIndex(&p->_head[i]));
        pagePtr[1] = pagePtr[0] + p->_length[i];
        p->_length[i] = 0;
        return;
    }
    pagePtr[0] = p->getPageAddr(i, p->getIndex(&p->_head[i]));
    pagePtr[1] = pagePtr[0] + p->_pageCapacity;
}

template <typename T>
static __global__ void page_popFree(DeviceListPool<T>* p, T** pagePtr) {
    assert(p->_freeHead != p->_freeRear);

    u_int32_t freeIndex = p->getIndex(&p->_freeHead);
    pagePtr[0] = p->_pool + p->_free[freeIndex] * p->_pageCapacity;
}


template <typename T>
static __global__ void page_pushFree(DeviceListPool<T>* p, u_int32_t pageId) {
    assert( (p->_freeRear + 1) % p->_listCapacity != p->_freeHead );

    u_int32_t freeIndex = p->getIndex(&p->_freeRear);
    p->_free[(freeIndex + 1) % p->_listCapacity] = pageId;
}

template <typename T>
class GPUListPool {

private:

    u_int32_t _numBuffer;
    u_int32_t _listCapacity;
    size_t _pageCapacity;
    u_int32_t _numPage;

    GPUBuffer<T> _pool;
    GPUBuffer<u_int32_t> _frame;

    GPUBuffer<T*> _rearPage;

    GPUBuffer<u_int32_t> _metadata;

    GPUBuffer<u_int32_t> _free;

    DeviceListPool<T> *_device;

    GPUBuffer<T*> _pagePtr;

    CPUBuffer<u_int32_t> _h_metadata;

    u_int32_t *_h_head;
    u_int32_t *_h_rear;
    u_int32_t *_h_length;

    CPUBuffer<T*> _h_pagePtr;
    int _gpuId;
public:
    GPUListPool(const GPUListPool<T>&) = delete;
    GPUListPool(GPUListPool<T>&&) = delete;

    GPUListPool& operator=(const GPUListPool<T>&) = delete;
    GPUListPool& operator=(GPUListPool<T>&&) = delete;

    GPUListPool(u_int32_t numBuffer, size_t pageCapacity, size_t numElement, int gpuId): \
        _numBuffer(numBuffer), _listCapacity((numElement + pageCapacity - 1) / pageCapacity + 2), _pageCapacity(pageCapacity), _numPage(_listCapacity + numBuffer * 2),\
        _pool(_numPage * pageCapacity, gpuId), _frame(numBuffer * _listCapacity, gpuId), _rearPage(numBuffer * 2, gpuId), _metadata(numBuffer * 3, gpuId), _free(_listCapacity, gpuId), \
        _h_metadata(numBuffer * 3), _pagePtr(2, gpuId), _h_pagePtr(2), _gpuId(gpuId)
    {
        DeviceListPool<T> host{_pool.ptr(), _frame.ptr(), _metadata.ptr(), _metadata.ptr() + numBuffer, _metadata.ptr() + numBuffer * 2, \
            _rearPage.ptr(), _free.ptr(), 0, _listCapacity - 1, _pageCapacity, _listCapacity, _numBuffer};
        cudaMalloc(&_device, sizeof(DeviceListPool<T>));
        gpuCall(cudaMemcpy(_device, &host, sizeof(DeviceListPool<T>), cudaMemcpyHostToDevice));

        _initDeviceListPool<<<_numPage / threadPerBlock + 1, threadPerBlock>>>(_device, numBuffer, _listCapacity);

        _h_metadata.cudaRegister();
        _h_pagePtr.cudaRegister();

        _h_length = _h_metadata.ptr();
        _h_head = _h_metadata.ptr() + numBuffer;
        _h_rear = _h_metadata.ptr() + numBuffer * 2;
    }

    ~GPUListPool() {
        cudaFree(_device);
    }

    DeviceListPool<T>* devicePtr() {
        return _device;
    }

    void insertFetchedPage(CUDAStream &stream) {
        page_insertFetched<T><<<1, threadPerBlock, 0, stream.get()>>>(_device);
    }

    void freeFirstPageAndInsertFetchedPage(u_int32_t i, CUDAStream &stream) {
        page_freeFirstAndInsertFetched<T><<<1, threadPerBlock, 0, stream.get()>>>(_device, i);
    }

    void copyLengthMetaData(CUDAStream &stream) {
        _h_metadata.fromAsync(_metadata, stream);
    }

    size_t totalBatch(u_int32_t i) {
        return ((_h_rear[i] + _listCapacity) - _h_head[i]) % _listCapacity + 1;
    }

    size_t totalLength(u_int32_t i) {
        return (totalBatch(i) - 1) * _pageCapacity + _h_length[i];
    }

    auto popFirstPage(u_int32_t i, CUDAStream &stream) {
        page_popFirst<T><<<1, 1, 0, stream.get()>>>(_device, i, _pagePtr.ptr());

        _h_pagePtr.fromAsync(_pagePtr, stream);

        stream.sync();

        return GPUBuffer<T>(_h_pagePtr[1] - _h_pagePtr[0], _h_pagePtr[0], _gpuId);
    }

    void evict(u_int32_t i, CUDAStream &stream, CPUBuffer<T> &page) {
        auto evicted = popFirstPage(i, stream);
        page.fromAsync(evicted, stream);

        pushFreePage(stream, evicted);
    }

    auto popFreePage(CUDAStream &stream) {
        page_popFree<T><<<1, 1, 0, stream.get()>>>(_device, _pagePtr.ptr());

        _h_pagePtr.fromAsync(_pagePtr, stream);
        stream.sync();

        return GPUBuffer<T>(_pageCapacity, _h_pagePtr[0], _gpuId);
    }

    void pushFreePage(CUDAStream &stream, GPUBuffer<T> &page) {
        page_pushFree<<<1, 1, 0, stream.get()>>>(_device, static_cast<u_int32_t>(page.ptr() - _pool.ptr()) / _pageCapacity);
    }

};
