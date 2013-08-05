#include <cuda_runtime.h>
#include <iostream>
#include "util.h"
#include "timer.h"

cuda_timer_pool timers;

cuda_timer::cuda_timer(bool start)
    : m_started(false)
    , m_elapsed(0)
    , m_start(NULL), m_stop(NULL)
{
    if(start)
        this->start();
}

cuda_timer::~cuda_timer()
{
    if(m_stop)
    {
        cudaEventDestroy(m_stop);
        check_cuda_error("Timer event destruction");
    }
    if(m_start)
    {
        cudaEventDestroy(m_start);
        check_cuda_error("Timer event destruction");
    }
}

void cuda_timer::start()
{
    if(m_started)
        stop();

    if(m_start == NULL)
    {
        cudaEventCreate(&m_start);
        check_cuda_error("Timer event creation");
    }
    if(m_stop == NULL)
    {
        cudaEventCreate(&m_stop);
        check_cuda_error("Timer event creation");
    }


    cudaEventRecord(m_start, 0);
    check_cuda_error("Event recording");
    m_elapsed = 0;
    m_started = true;
}

void cuda_timer::stop()
{
    if(!m_started)
        return;

    cudaEventRecord(m_stop, 0);
    check_cuda_error("Event recording");
    m_started = false;
}

float cuda_timer::elapsed()
{ 
    if(m_elapsed == 0)
    {
        stop();

        cudaEventSynchronize(m_stop);
        check_cuda_error("Event synchronize");
        cudaEventElapsedTime(&m_elapsed, m_start, m_stop);
        check_cuda_error("Event elapsed time");
    }

    return m_elapsed; 
}

#if !DISABLE_TIMER_POOL
cuda_timer &cuda_timer_pool::add(const std::string &label, bool start)
{
    cuda_timer *timer = new cuda_timer(false);
    m_timers.push_back(std::make_pair(timer, label));

    if(start)
        timer->start();
    return *timer;
}

void cuda_timer_pool::flush()
{
    for(timer_list::iterator it=m_timers.begin(); it!=m_timers.end(); ++it)
    {
        std::cout << it->second << ": ";
        if(!it->first->is_stopped())
            std::cout << "FORCED STOP - ";
        std::cout << it->first->elapsed() << std::endl;

        delete it->first;
    }

    m_timers.clear();
}
#endif

cuda_timer_pool::~cuda_timer_pool()
{
    flush();
}
