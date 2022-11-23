#pragma once

#include <chrono>
#include <vector>

#include "utils/Event.h"

class Timer
{
private:
    friend class TimerGuard;

    std::vector<std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point>> time;

    void push(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point stop) {
        time.emplace_back(start, stop);
    }

public:
    float elapsed() {
        size_t sum = 0;
        for (auto& t: time) {
            sum += std::chrono::duration_cast<std::chrono::nanoseconds>(t.second - t.first).count();
        }

        return (sum + 0.0) / 1000.0 / 1000.0;
    }
};

class TimerGuard
{
private:
    std::chrono::high_resolution_clock::time_point _start;
    Timer &_timer;

public:
    TimerGuard(Timer &timer): _timer(timer) {
        _start = std::chrono::high_resolution_clock::now();
    }

    ~TimerGuard() {
        std::chrono::high_resolution_clock::time_point stop = std::chrono::high_resolution_clock::now();
        _timer.push(_start, stop);
    }
};



class GPUTimer
{
private:
    friend class GPUTimerGuard;

    std::vector<std::unique_ptr<CUDAEvent>> starts;
    std::vector<std::unique_ptr<CUDAEvent>> stops;

    CUDAStream &_stream;

    void start() {
        starts.emplace_back(std::make_unique<CUDAEvent>(_stream));
    }

    void stop() {
        stops.emplace_back(std::make_unique<CUDAEvent>(_stream));
    }

public:
    GPUTimer(CUDAStream &stream): _stream(stream) {}

    float elapsed() {
        float sum = 0;
        for (int i = 0; i < starts.size(); i++) {
            float t;
            cudaEventElapsedTime(&t, starts[i]->get(), stops[i]->get());

            sum += t;
        }

        return sum;
    }

    size_t calls() {
        return stops.size();
    }
};

class GPUTimerGuard
{
private:
    GPUTimer &_timer;

public:
    GPUTimerGuard(GPUTimer &timer): _timer(timer) {
        #ifndef NO_EVENT_TIMER
            timer.start();
        #endif
    }

    ~GPUTimerGuard() {
        #ifndef NO_EVENT_TIMER
            _timer.stop();
        #endif
    }
};

class Stopwatch
{
private:
    std::chrono::high_resolution_clock::time_point startTime, stopTime;
    size_t totalTime;

public:
    explicit Stopwatch(bool run = false): totalTime(0) {
        if (run) {
            start();
        }
    }

    void start() { 
        startTime = stopTime = std::chrono::high_resolution_clock::now(); 
    }

    void stop() { 
        stopTime = std::chrono::high_resolution_clock::now(); 
        totalTime += std::chrono::duration_cast<std::chrono::nanoseconds>(stopTime - startTime).count();
    }

    size_t total() const {
        return totalTime;
    }
};
