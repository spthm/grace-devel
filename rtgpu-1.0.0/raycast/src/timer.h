#ifndef TIMER_H
#define TIMER_H

#include <string>
#include <list>

#define DISABLE_TIMER_POOL 1

class cuda_timer
{
public:
    cuda_timer(bool start = true);
    ~cuda_timer();

    void start();
    void stop();
    float elapsed();
    bool is_stopped() const { return !m_started; }

private:
    cuda_timer(const cuda_timer &);
    cuda_timer &operator=(const cuda_timer &);

    cudaEvent_t m_start, m_stop;
    float m_elapsed;
    bool m_started;
};

class scoped_timer_stop
{
public:
    scoped_timer_stop(cuda_timer &timer) : m_timer(&timer) {}
    ~scoped_timer_stop() { stop(); }

    void stop() { m_timer->stop(); }

private:
    cuda_timer *m_timer;
};

class cuda_timer_pool
{
public:
    ~cuda_timer_pool();

#if DISABLE_TIMER_POOL
    cuda_timer &add(const std::string &label, bool start = true)
    {
        static cuda_timer t(false);
        return t;
    }
    void flush() {}
#else
    cuda_timer &add(const std::string &label, bool start = true);
    void flush();
#endif

private:
    typedef std::list<std::pair<cuda_timer*, std::string> > timer_list;
    timer_list m_timers;
};

extern cuda_timer_pool timers;

#endif
