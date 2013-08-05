#ifndef RTGPU_CANVAS_H
#define RTGPU_CANVAS_H

#include "model.h"
#include "light.h"

class canvas : public Fl_Gl_Window
{
public:
    canvas(int x, int y, int w, int h);
    ~canvas();

    virtual void draw();
    virtual int handle(int ev);

private:
    void resize_buffer(const s3d::r2::usize &size);
    void raytrace();
    void copy_buffer_to_texture();
    optixu::Context m_ctx;

    bool m_glOK;

    boost::optional<s3d::r2::point> m_model_drag_press_pos,
				    m_light_drag_press_pos,
				    m_view_drag_press_pos;
    s3d::r4::unit_quaternion m_model_rot, m_model_drag_rot,
			     m_light_rot, m_light_drag_rot,
			     m_view_rot, m_view_drag_rot;
    float m_dist;
     
    GLuint m_fbo;

    s3d::r2::usize m_canvas_size;
    optixu::Buffer m_output_buffer, m_lights_buffer;

    optixu::Buffer m_ray_buffer;

    std::vector<shade::point_light> m_point_lights;
    void update_lights();

    GLuint m_tex_display;

    float m_start_time;

    void init_gl();
    void destroy_gl();

    optixu::GeometryGroup create_geometry();
};

#endif
