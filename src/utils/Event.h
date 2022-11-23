#pragma once

class CUDAEvent;

class CUDAStream {
private:
    cudaStream_t _stream;
    int _deviceId;

public:
    friend class CUDAEvent;

    CUDAStream(const CUDAStream&) = delete;
    CUDAStream(CUDAStream&&) = delete;

    CUDAStream& operator=(const CUDAStream&) = delete;
    CUDAStream& operator=(CUDAStream&&) = delete;

    CUDAStream(int deviceId);

    ~CUDAStream();

    void sync() const;

    void wait(CUDAEvent &event) const;

    auto record() const;

    bool isBusy() const;

    cudaStream_t& get();
};

class CUDAEvent {

private:
    cudaEvent_t _event;

public:
    friend class CUDAStream;

    CUDAEvent() {
        cudaEventCreate(&_event);
    }

    CUDAEvent(CUDAStream &stream) {
        cudaEventCreate(&_event);
        cudaEventRecord(_event, stream._stream);
    }

    ~CUDAEvent() {
        cudaEventDestroy(_event);
    }

    void sync() const {
        cudaEventSynchronize(_event);
    }

    bool done() const {
        return cudaEventQuery(_event);
    }

    cudaEvent_t& get() {
        return _event;
    }

    CUDAEvent(const CUDAEvent&) = delete;
    // always wrap this with smart pointer
    CUDAEvent(CUDAEvent&&) = default;

    CUDAEvent& operator=(const CUDAEvent&) = delete;
    CUDAEvent& operator=(CUDAEvent&&) = delete;
};

CUDAStream::CUDAStream(int deviceId): _deviceId(deviceId) {
    cudaSetDevice(deviceId);
    cudaStreamCreate(&_stream);
}

CUDAStream::~CUDAStream() {
    cudaStreamDestroy(_stream);
}

void CUDAStream::sync() const {
    cudaStreamSynchronize(_stream);
}

void CUDAStream::wait(CUDAEvent &event) const {
    cudaStreamWaitEvent(_stream, event._event, 0);
}

auto CUDAStream::record() const {
    cudaSetDevice(_deviceId);
    CUDAEvent event;
    cudaEventRecord(event._event, _stream);

    return event;
}

bool CUDAStream::isBusy() const {
    return cudaStreamQuery(_stream);
}

cudaStream_t& CUDAStream::get() {
    return _stream;
}
