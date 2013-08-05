#include "pch.h"
#include "symbol.h"
#include "light.h"
#include "canvas.h"
#include "util.h"
#include "raytrace.h"
#include "bvh.h"

#define SPONZA 1
#define TEAPOT 2

#define MODEL SPONZA

using namespace s3d;

canvas::canvas(int x, int y, int w, int h) /*{{{*/
    : Fl_Gl_Window(x,y,w,h)
    , m_glOK(false)
    , m_start_time(get_time())
    , m_canvas_size(0,0)
    , m_tex_display(0)
    , m_fbo(0)
    , m_light_pos_dirty(true)
    , m_scene_dirty(true)
    , m_bvh_height(0)
#if MODEL == TEAPOT
    , m_dist(1.5)
    , m_view_rot(-0.895204,-0.257244,-0.332546,-0.147813)
#elif MODEL == SPONZA
    , m_view_rot(-0.842163,-0.0122688,-0.539028,-0.00777033)
    , m_dist(0.377354)
#endif
//   , m_view_rot(to_quaternion(r3::y_axis, M_PI/4-M_PI/16))
{
    cudaGLSetGLDevice(0);
    m_cuda_output_buffer = NULL;
}/*}}}*/

canvas::~canvas()/*{{{*/
{
    if(m_glOK)
	destroy_gl();
}/*}}}*/

int canvas::handle(int ev)/*{{{*/
{
    r2::point pos(Fl::event_x(), h()-Fl::event_y());

    switch(ev)
    {
    case FL_PUSH:
	switch(Fl::event_button())
	{
	case FL_LEFT_MOUSE:
	    assert(!m_view_drag_press_pos);
	    m_view_drag_press_pos = pos;
	    return 1;
	case FL_MIDDLE_MOUSE:
	    assert(!m_model_drag_press_pos);
	    m_model_drag_press_pos = pos;
	    return 1;
	case FL_RIGHT_MOUSE:
	    assert(!m_light_drag_press_pos);
	    m_light_drag_press_pos = pos;
	    return 1;
	}
	break;
    case FL_RELEASE:
	switch(Fl::event_button())
	{
	case FL_LEFT_MOUSE:
	    if(m_view_drag_press_pos)
	    {
		m_view_rot = m_view_drag_rot*m_view_rot;
		m_view_drag_rot = r4::unit_quaternion();
		m_view_drag_press_pos.reset();
		return 1;
	    }
	    break;
	case FL_MIDDLE_MOUSE:
	    if(m_model_drag_press_pos)
	    {
		m_model_rot = m_model_drag_rot*m_model_rot;
		m_model_drag_rot = r4::unit_quaternion();
		m_model_drag_press_pos.reset();
		return 1;
	    }
	    break;
	case FL_RIGHT_MOUSE:
	    if(m_light_drag_press_pos)
	    {
		m_light_rot = m_light_drag_rot * m_light_rot;
		m_light_drag_rot = r4::unit_quaternion();
		m_light_drag_press_pos.reset();
		return 1;
	    }
	    break;
	}
	break;
    case FL_DRAG:
	if(Fl::event_button1() && m_view_drag_press_pos)
	{
	    m_view_drag_rot = trackball(*m_view_drag_press_pos, pos,
					{w()/2.0f,h()/2.0f}, w()/3);
            m_scene_dirty = true;
	    return 1;
	}
	if(Fl::event_button2() && m_model_drag_press_pos)
	{
	    auto rot = trackball(*m_model_drag_press_pos, pos,
				 {w()/2.0f,h()/2.0f}, w()/3);

	    m_model_drag_rot = inv(m_view_rot)*rot*m_view_rot;
            m_scene_dirty = true;
	    return 1;
	}
	if(Fl::event_button3() && m_light_drag_press_pos)
	{
	    auto rot = trackball(*m_light_drag_press_pos, pos,
				 {w()/2.0f,h()/2.0f}, w()/3);
	    m_light_drag_rot = inv(m_view_rot)*rot*m_view_rot;
            m_light_pos_dirty = true;
	    return 1;
	}
	break;
    case FL_MOUSEWHEEL:
	if(Fl::event_dy() > 0)
	{
	    m_dist *= 1.05;
            m_scene_dirty = true;
	    return 1;
	}
	else if(Fl::event_dy() < 0)
	{
	    m_dist *= 0.95;
            m_scene_dirty = true;
	    return 1;
	}
	break;
    }

    return Fl_Gl_Window::handle(ev);
}/*}}}*/

void canvas::resize_buffer(const r2::usize &size)/*{{{*/
{
    m_canvas_size = size;

    cudaError_t err = cudaGraphicsUnregisterResource(m_cuda_output_buffer);
    if(err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    // modifica tamanho do PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size.w*size.h*sizeof(float4),
		 NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // modifica tamanho da textura

    glBindTexture(GL_TEXTURE_2D, m_tex_display);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 
                 m_canvas_size.w, m_canvas_size.h, 0, 
                 GL_RGBA, GL_FLOAT, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);

    err = cudaGraphicsGLRegisterBuffer(&m_cuda_output_buffer, m_pbo, 
                                 cudaGraphicsMapFlagsWriteDiscard);

    if(err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));
}/*}}}*/
void canvas::copy_buffer_to_texture()/*{{{*/
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBindTexture(GL_TEXTURE_2D, m_tex_display);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    glTexSubImage2D(GL_TEXTURE_2D, 0,
                 0,0,m_canvas_size.w, m_canvas_size.h,
                 GL_RGBA, GL_FLOAT, NULL);

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}/*}}}*/

void canvas::init_gl()/*{{{*/
{
    {
        GLenum err = glewInit();
        if(err != GLEW_OK)
            throw std::runtime_error((char *)glewGetErrorString(err));
    }

    // cria PBO
    glGenBuffers(1, &m_pbo);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 1*1*4*sizeof(GLfloat),
		 NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    cudaError_t err = 
        cudaGraphicsGLRegisterBuffer(&m_cuda_output_buffer, m_pbo, 
                                 cudaGraphicsMapFlagsWriteDiscard);

    if(err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    // cria textura 
    glGenTextures(1, &m_tex_display);
    glBindTexture(GL_TEXTURE_2D, m_tex_display);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, 0);

    check_glerror();

    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
    glFramebufferTexture(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,
			 m_tex_display,0);
    glReadBuffer(GL_COLOR_ATTACHMENT0);

    check_glerror();

    // ajusta tamanho das coisas (p/ estado interno ficar OK)
    resize_buffer({1,1});

#if MODEL == SPONZA
    shade::PointLight light({1,1,1}, 1, {0,0,0});
#else
    shade::PointLight light({1,1,1}, 1, {2,2,0});
#endif
    light.casts_shadow = true;
    m_point_lights.push_back(light);

    update_lights();

    m_spheres.emplace_back(make_float3(-0.5, 0, 0.5), 0.3);
    m_spheres.emplace_back(make_float3(0.5, 0, 0.5), 0.3);

    m_spheres.emplace_back(make_float3(-0.5,0,-0.5),0.5);
    m_spheres.emplace_back(make_float3(0.5,0,-0.5),0.5);

    update_scene();

    // configura OpenGL
    glClearColor(0,0,0,1);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();


    m_glOK = true;
}/*}}}*/
void canvas::destroy_gl()/*{{{*/
{
    glDeleteBuffers(1, &m_pbo);

    glDeleteFramebuffers(1, &m_fbo);

    glDeleteTextures(1, &m_tex_display);
}/*}}}*/

void canvas::update_lights()/*{{{*/
{
    copy_to_symbol("lights","num_lights",m_point_lights);
}/*}}}*/
void canvas::update_scene()/*{{{*/
{
#if MODEL == SPONZA
    auto model = load_model("sponza.ply");
    float scale = 1;
#elif MODEL == TEAPOT
    auto model = load_model("teapot.ply");
    float scale = 2;
#endif

    auto bbox = bounding_box(model);
    int maxdim = maximum_extent(bbox);

    for(unsigned i=0; i<model.positions.size(); ++i)
        model.positions[i] = (model.positions[i] - centroid(bbox))/
                             bbox.size[maxdim]*scale;

    // must remove degenerate triangles
    for(int i=0; i<model.faces.size(); ++i)
    {
        auto face = make_face_view(model.faces[i], model.positions);
        r3::dvector v0(face[0]), v1(face[1]), v2(face[2]);

        using std::abs;

        r3::dvector c = cross(v0-v2, v1-v2);
        if(sqrnorm(c) == 0)
        {
            swap(model.faces[i], model.faces.back());
            model.faces.resize(model.faces.size()-1);
            --i;
        }
    }

    m_bvh = make_linear_8bvh(model, &m_bvh_height);

    m_model.xform.clear();
    m_model.xform.reserve(model.faces.size()*3);

    m_model.normals.clear();
    m_model.normals.reserve(model.faces.size()*3);

    for(int i=0; i<model.faces.size(); ++i)
    {
        auto face = make_face_view(model.faces[i], model.positions);

        auto normal = make_face_view(model.faces[i], model.normals);

        r3::dvector off(face[2]);

        r3::dvector m0 = face[0]-off,
                    m1 = face[1]-off,
                    c = cross(m0,m1);

        r3::vector m2 = unit(c) - off;

        math::AffineTransform<double,3> xform({{m0.x, m1.x, m2.x, off.x},
                                               {m0.y, m1.y, m2.y, off.y},
                                               {m0.z, m1.z, m2.z, off.z}});
        xform = inv(xform);

        for(int j=0; j<3; ++j)
        {
            m_model.xform.push_back(
                make_float4(xform[j].x, xform[j].y, xform[j].z, xform[j].w));

            m_model.normals.push_back(
                make_float3(normal[j].x, normal[j].y, normal[j].z));
        }
    }

    assert(m_model.xform.size() == model.faces.size()*3);

}/*}}}*/

void canvas::raytrace()
{
#if 0
    if(m_light_pos_dirty)
    {
        r3::point light_pos(2,2,2);

        auto light_rot = m_light_drag_rot*m_light_rot;

        light_pos = rot(light_pos, light_rot);

     //   light_pos = {-0.728125,2.98935,-1.59172};

        m_point_lights[0].pos = {light_pos.x, light_pos.y, light_pos.z};

        update_lights();

        m_light_pos_dirty = false;
    }
#endif

//    std::cout << "LIGHT: " << light_pos << std::endl;

    r3::frustum frustum;
#if 1
    frustum.pos = {0,0,m_dist};
    frustum.up = r3::y_axis;
    frustum.axis = -r3::z_axis;
#endif

    frustum.aspect = float(w())/h();
    frustum.fov = rad(60); // hfov

    auto view_rot = inv(m_view_drag_rot*m_view_rot);

#if 1
    frustum.pos = rot(frustum.pos, view_rot);
    frustum.up = rot(frustum.up, view_rot);
    frustum.axis = rot(frustum.axis, view_rot);
#endif

#if 0
    std::cout << "CAM POS: " << frustum.pos << std::endl;
    std::cout << "CAM UP: " << frustum.up << std::endl;
    std::cout << "CAM AXIS: " << frustum.axis << std::endl;
#endif

    r3::vector w = frustum.axis,
	       u = unit(cross(w, frustum.up))*fov_horiz(frustum),
	       v = frustum.up*fov_vert(frustum);

    r3::matrix invM = inv(r3::matrix({u,v,w}));

    float3 cuda_u = {u.x, u.y, u.z};
    float3 cuda_v = {v.x, v.y, v.z};
    float3 cuda_w = {w.x, w.y, w.z};

    float3 cuda_invu = {invM[0][0], invM[0][1], invM[0][2]};
    float3 cuda_invv = {invM[1][0], invM[1][1], invM[1][2]};
    float3 cuda_invw = {invM[2][0], invM[2][1], invM[2][2]};

    float3 cuda_eye = {frustum.pos.x, frustum.pos.y, frustum.pos.z};

    cuda_trace(cuda_u, cuda_v, cuda_w, cuda_invu, cuda_invv, cuda_invw,
               cuda_eye,
               m_model, m_bvh, m_bvh_height,
               m_cuda_output_buffer, 
               m_canvas_size.w, m_canvas_size.h, m_scene_dirty);

    m_scene_dirty = false;
}

void canvas::draw()
{
    if(!m_glOK)
	init_gl();

    if(m_canvas_size != r2::isize(w(), h()))
	resize_buffer({w(), h()});

    raytrace();
    copy_buffer_to_texture();

    glBlitFramebuffer(0,0,m_canvas_size.w-1, m_canvas_size.h-1,
		      0,0,m_canvas_size.w-1, m_canvas_size.h-1,
		      GL_COLOR_BUFFER_BIT, GL_NEAREST);
}
