#pragma once

#include <assert.h>

#include "utils/Event.h"
#include "utils/GPUCall.h"

template <typename T>
class CPUBuffer;

template <typename T>
class GPUBuffer;

template <typename T>
class DiskBuffer
{

friend class CPUBuffer<T>;

private:
    FILE *_fp;
    size_t _length;
    size_t _byteOffset;

public:
    DiskBuffer(FILE *fp, size_t byteOffset, size_t length): \
        _fp(fp), _length(length), _byteOffset(byteOffset) {}

    void to(CPUBuffer<T> &buffer) {
        fseek(_fp, _byteOffset, SEEK_SET);
        fread(buffer.ptr(), sizeof(T), this->_length, _fp);

        buffer._length = this->_length;
    }

    void from(CPUBuffer<T> &buffer) {
        fseek(_fp, _byteOffset, SEEK_SET);
        fwrite(buffer.ptr(), sizeof(T), this->_length, _fp);

        this->_length = buffer.length();
    }
};

template <typename T>
class CPUBuffer
{

friend class DiskBuffer<T>;
friend class GPUBuffer<T>;

private:
    T* _ptr;
    bool _owner;

    size_t _length;

    bool _cudaPinned{false};

public:
    CPUBuffer(const CPUBuffer<T>&) = delete;
    CPUBuffer(CPUBuffer<T>&& other) {
        _ptr = other._ptr;
        _owner = other._owner;
        _length = other._length;
        _cudaPinned = other._cudaPinned;

        other._owner = false;
    }

    CPUBuffer& operator=(const CPUBuffer<T>&) = delete;
    CPUBuffer& operator=(CPUBuffer<T>&&) = delete;

    CPUBuffer(size_t length): _owner(true), _length(length)
    {
        _ptr = new T[length];
    }

    CPUBuffer(size_t length, T* ptr): _ptr(ptr), _owner(false), _length(length) {}

    ~CPUBuffer() {
        if (_owner)
            delete[] _ptr;
    }

    template <typename S>
    CPUBuffer(CPUBuffer<S> &buffer): _ptr((T *)buffer.ptr()), _owner(false), _length(buffer.sizeInByte() / sizeof(T)) {
        assert(buffer.sizeInByte() % sizeof(S) == 0);
    }

    size_t length() {
        return _length;
    }

    size_t sizeInByte() {
        return length() * sizeof(T);
    }

    void to(DiskBuffer<T> &buffer) {
        buffer.from(*this);
    }

    void from(DiskBuffer<T> &buffer) {
        buffer.to(*this);
    }

    void to(CPUBuffer<T> &buffer) {
        std::memcpy(buffer.ptr(), _ptr, this->_length * sizeof(T));
    }

    void from(CPUBuffer<T> &buffer) {
        buffer.to(*this);
    }

    void to(GPUBuffer<T> &buffer) {
        buffer.from(*this);
    }
    
    void from(GPUBuffer<T> &buffer) {
        buffer.to(*this);
    }

    void toAsync(GPUBuffer<T> &buffer, CUDAStream &stream) {
        buffer.fromAsync(*this, stream);
    }

    void fromAsync(GPUBuffer<T> &buffer, CUDAStream &stream) {
        buffer.toAsync(*this, stream);
    }

    void cudaRegister() {
        if (!_cudaPinned && _owner)
            cudaHostRegister((void *)_ptr, _length * sizeof(T), 0);
        _cudaPinned = true;
    }

    T& operator[](size_t i) const {
        return _ptr[i];
    }
    
    T* ptr() const {
        return _ptr;
    }

    void clear() {
        this->_length = 0;
    }
};

template <typename T>
class GPUBuffer
{

friend class CPUBuffer<T>;

private:
    T *_ptr;
    bool _owner;
    
    size_t _length;

    int _gpuId;

public:
    GPUBuffer(const GPUBuffer<T>&) = delete;
    GPUBuffer(GPUBuffer<T>&& other) {
        _ptr = other._ptr;
        _owner = other._owner;
        _length = other._length;
        _gpuId = other._gpuId;

        other._owner = false;
    }

    GPUBuffer& operator=(const GPUBuffer<T>&) = delete;
    GPUBuffer& operator=(GPUBuffer<T>&&) = delete;

    GPUBuffer(size_t length, int gpuId): _owner(true), _gpuId(gpuId)
    {
        cudaSetDevice(gpuId);
        gpuCall(cudaMalloc(&_ptr, sizeof(T) * length));
        _length = length;
    }

    GPUBuffer(size_t length, T* ptr, int gpuId): _ptr(ptr), _owner(false), _length(length), _gpuId(gpuId) {}

    ~GPUBuffer() {
        if (_owner)
            gpuCall(cudaFree(_ptr));
    }

    T* ptr() const {
        return _ptr;
    }

    size_t length() const {
        return _length;
    }

    size_t sizeInByte() {
        return length() * sizeof(T);
    }

    void from(CPUBuffer<T> &buffer) {
        gpuCall(cudaMemcpy(_ptr, buffer.ptr(), sizeof(T) * buffer.length(), cudaMemcpyHostToDevice));

        #ifdef HALF_PCIE_BANDWIDTH
        gpuCall(cudaMemcpy(_ptr, buffer.ptr(), sizeof(T) * buffer.length(), cudaMemcpyHostToDevice));
        #endif

        _length = buffer.length();
    }

    void fromAsync(CPUBuffer<T> &buffer, CUDAStream &stream) {
        gpuCall(cudaMemcpyAsync(_ptr, buffer.ptr(), sizeof(T) * buffer.length(), cudaMemcpyHostToDevice, stream.get()));

        #ifdef HALF_PCIE_BANDWIDTH
        gpuCall(cudaMemcpyAsync(_ptr, buffer.ptr(), sizeof(T) * buffer.length(), cudaMemcpyHostToDevice, stream.get()));
        #endif

        _length = buffer.length();
    }

    void to(CPUBuffer<T> &buffer) {
        gpuCall(cudaMemcpy(buffer.ptr(), _ptr, sizeof(T) * _length, cudaMemcpyDeviceToHost));

        #ifdef HALF_PCIE_BANDWIDTH
        gpuCall(cudaMemcpy(buffer.ptr(), _ptr, sizeof(T) * _length, cudaMemcpyDeviceToHost));
        #endif

        buffer._length = _length;
    }

    void toAsync(CPUBuffer<T> &buffer, CUDAStream &stream) {
        gpuCall(cudaMemcpyAsync(buffer.ptr(), _ptr, sizeof(T) * _length, cudaMemcpyDeviceToHost, stream.get()));

        #ifdef HALF_PCIE_BANDWIDTH
        gpuCall(cudaMemcpyAsync(buffer.ptr(), _ptr, sizeof(T) * _length, cudaMemcpyDeviceToHost, stream.get()));
        #endif

        buffer._length = _length;
    }

    void from(GPUBuffer<T> &buffer) {
        gpuCall(cudaMemcpy(_ptr, buffer.ptr(), sizeof(T) * buffer.length(), cudaMemcpyDeviceToDevice));

        #ifdef HALF_PCIE_BANDWIDTH
        gpuCall(cudaMemcpy(_ptr, buffer.ptr(), sizeof(T) * buffer.length(), cudaMemcpyDeviceToDevice));
        #endif

        _length = buffer.length();
    }

    void fromAsync(GPUBuffer<T> &buffer, CUDAStream &stream) {
        gpuCall(cudaMemcpyAsync(_ptr, buffer.ptr(), sizeof(T) * buffer.length(), cudaMemcpyDeviceToDevice, stream.get()));

        #ifdef HALF_PCIE_BANDWIDTH
        gpuCall(cudaMemcpyAsync(_ptr, buffer.ptr(), sizeof(T) * buffer.length(), cudaMemcpyDeviceToDevice, stream.get()));
        #endif

        _length = buffer.length();
    }

    void to(GPUBuffer<T> &buffer) {
        buffer.from(*this);
    }

    void toAsync(GPUBuffer<T> &buffer, CUDAStream &stream) {
        buffer.fromAsync(*this, stream);
    }
};
