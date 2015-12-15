#pragma once

#include <algorithm>

class CUDATimer
{
private:
    cudaEvent_t start_, stop_, split_;

public:
    CUDATimer()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
        cudaEventCreate(&split_);
    }

    ~CUDATimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
        cudaEventDestroy(split_);
    }

    // Records the start point of the timer.
    void start()
    {
        // Assigning one as the copy-assignment of the other seems to result in
        // incorrect behaviour.
        cudaEventRecord(start_);
        cudaEventRecord(split_);
    }

    // Returns the time in ms since .start() was last called.
    float elapsed()
    {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);

        float elapsed;
        cudaEventElapsedTime(&elapsed, start_, stop_);

        return elapsed;
    }

    // Returns the time in ms since .split() was last called, or, if .split()
    // has not been called, the time since .start() was called.
    float split()
    {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);

        float elapsed;
        cudaEventElapsedTime(&elapsed, split_, stop_);
        std::swap(split_, stop_);

        return elapsed;
    }
};
