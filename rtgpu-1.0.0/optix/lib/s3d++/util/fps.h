#ifndef S3D_UTIL_FPS_H
#define S3D_UTIL_FPS_H

namespace s3d
{

class fps_counter
{
public:
    fps_counter() 
		: m_frames(0), m_prev_time(0), m_prev_print_time(0), m_fps(0) {}

    bool update(double dt);
    double value() const { return m_fps; }
    bool print(double dt);
private:
    unsigned m_frames;
    double m_prev_time,
    	   m_prev_print_time, m_fps;
};

}

#endif
