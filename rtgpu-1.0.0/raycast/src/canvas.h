#ifndef RTGPU_CANVAS_H
#define RTGPU_CANVAS_H

#include "model.h"
#include "light.h"
#include "types.h"

struct linear_8bvh_node;

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

    bool m_glOK;

    boost::optional<s3d::r2::point> m_model_drag_press_pos,
				    m_light_drag_press_pos,
				    m_view_drag_press_pos;
    s3d::r4::unit_quaternion m_model_rot, m_model_drag_rot,
			     m_light_rot, m_light_drag_rot,
			     m_view_rot, m_view_drag_rot;
    bool m_light_pos_dirty, m_scene_dirty;

    float m_dist;
     
    GLuint m_fbo, m_pbo;
    cudaGraphicsResource *m_cuda_output_buffer;

    Mesh m_model;
    std::vector<linear_8bvh_node> m_bvh;
    size_t m_bvh_height;

    s3d::r2::usize m_canvas_size;

    std::vector<shade::PointLight> m_point_lights;
    void update_lights();

    std::vector<Sphere> m_spheres;
    void update_scene();

    GLuint m_tex_display;

    float m_start_time;

    void init_gl();
    void destroy_gl();
};

#endif
