#include "pch.h"
#include "time.h"
#include "fps.h"

namespace s3d
{

bool fps_counter::update(double dt)
{
    double agora = get_time();

    if(m_prev_time == 0)
    	m_prev_time = agora;

    ++m_frames;

    double elapsed = agora-m_prev_time;

    if(elapsed >= dt)
    {
    	m_fps = m_frames/elapsed;
    	m_frames = 0;
    	m_prev_time = agora;
    	return true;
    }
    else
    	return false;
}

bool fps_counter::print(double dt)
{
    double agora = get_time();
    if(m_prev_print_time == 0)
    	m_prev_print_time = agora;

    double elapsed = agora-m_prev_print_time;

    if(elapsed >= dt)
    {
    	m_prev_print_time = agora;
    	return true;
    }
    else
    	return false;
}

}
