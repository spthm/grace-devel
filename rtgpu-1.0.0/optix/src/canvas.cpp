#include "pch.h"
#include "light.h"
#include "canvas.h"
#include "util.h"

#define SPONZA 1
#define TEAPOT 2

#define MODEL SPONZA

#define CUDA_PREFIX "cuda_compile_ptx_generated_"

const char *PINHOLE_CAMERA_PTX = CUDA_PREFIX"pinhole_camera.cu.ptx",
           *SPHERE_PTX = CUDA_PREFIX"sphere.cu.ptx",
           *BOX_PTX = CUDA_PREFIX"box.cu.ptx",
           *MESH_PTX = CUDA_PREFIX"triangle_mesh.cu.ptx",
           *PARALLELOGRAM_PTX = CUDA_PREFIX"parallelogram.cu.ptx",
           *FLAT_MATERIAL_PTX = CUDA_PREFIX"flat_shading.cu.ptx",
           *PLASTIC_MATERIAL_PTX = CUDA_PREFIX"plastic.cu.ptx",
           *MATTE_MATERIAL_PTX = CUDA_PREFIX"matte.cu.ptx";

using namespace s3d;
namespace opt = optixu;

struct ray_info
{
    float3 origin;
    float2 dir;
};

canvas::canvas(int x, int y, int w, int h) /*{{{*/
    : Fl_Gl_Window(x,y,w,h)
    , m_glOK(false)
    , m_start_time(get_time())
    , m_canvas_size(0,0)
    , m_tex_display(0)
    , m_fbo(0)
#if MODEL == SPONZA
    , m_view_rot(-0.842163,-0.0122688,-0.539028,-0.00777033)
    , m_dist(0.377354)
#elif MODEL == TEAPOT
    , m_dist(1.5)
    , m_view_rot(-0.895204,-0.257244,-0.332546,-0.147813)
    , m_light_rot(0.36577,-0.0601298,-0.440248,0.817789)
#endif
{

    m_ctx = opt::Context::create();
    m_ctx->setRayTypeCount(2);
    m_ctx->setEntryPointCount(1);
    m_ctx->setStackSize(8000);

    m_ctx["max_depth"]->setInt(10);
    m_ctx["scene_epsilon"]->setFloat(1e-3);

    opt::Program ray_gen_program 
	= m_ctx->createProgramFromPTXFile(PINHOLE_CAMERA_PTX, "pinhole_camera");
    m_ctx->setRayGenerationProgram(0, ray_gen_program);

    opt::Program miss_program
	= m_ctx->createProgramFromPTXFile(PINHOLE_CAMERA_PTX, "miss");
    m_ctx->setMissProgram(0, miss_program);

    opt::Program exception_program
	= m_ctx->createProgramFromPTXFile(PINHOLE_CAMERA_PTX, "exception");
    m_ctx->setExceptionProgram(0, exception_program);
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
	    return 1;
	}
	if(Fl::event_button2() && m_model_drag_press_pos)
	{
	    auto rot = trackball(*m_model_drag_press_pos, pos,
				 {w()/2.0f,h()/2.0f}, w()/3);

	    m_model_drag_rot = inv(m_view_rot)*rot*m_view_rot;
	    return 1;
	}
	if(Fl::event_button3() && m_light_drag_press_pos)
	{
	    auto rot = trackball(*m_light_drag_press_pos, pos,
				 {w()/2.0f,h()/2.0f}, w()/3);
	    m_light_drag_rot = inv(m_view_rot)*rot*m_view_rot;
	    return 1;
	}
	break;
    case FL_MOUSEWHEEL:
	if(Fl::event_dy() > 0)
	{
	    m_dist *= 1.05;
	    return 1;
	}
	else if(Fl::event_dy() < 0)
	{
	    m_dist *= 0.95;
	    return 1;
	}
	break;
    }

    return Fl_Gl_Window::handle(ev);
}/*}}}*/

void canvas::resize_buffer(const r2::usize &size)/*{{{*/
{
    m_canvas_size = size;

    // Modifica tamanho do buffer
    m_output_buffer->unregisterGLBuffer();
    m_output_buffer->setSize(size.w, size.h);

    // modifica tamanho do PBO
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_output_buffer->getGLBOId());
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 
		 size.w*size.h*m_output_buffer->getElementSize(),
		 NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // modifica tamanho da textura

    glBindTexture(GL_TEXTURE_2D, m_tex_display);

    switch(m_output_buffer->getFormat())
    {
    case RT_FORMAT_UNSIGNED_BYTE4:
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 
		     m_canvas_size.w, m_canvas_size.h, 0, 
		     GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	break;
    case RT_FORMAT_FLOAT4:
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 
		     m_canvas_size.w, m_canvas_size.h, 0, 
		     GL_RGBA, GL_FLOAT, NULL);
	break;
    case RT_FORMAT_FLOAT3:
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 
		     m_canvas_size.w, m_canvas_size.h, 0, 
		     GL_RGB, GL_FLOAT, NULL);
	break;
    case RT_FORMAT_FLOAT:
	glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, 
		     m_canvas_size.w, m_canvas_size.h, 0, 
		     GL_LUMINANCE, GL_FLOAT, NULL);
	break;
    default:
	assert(false && "Unknown buffer format");
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    m_output_buffer->registerGLBuffer();
}/*}}}*/
#if 0
void canvas::copy_buffer_to_texture()/*{{{*/
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_output_buffer->getGLBOId());
    glBindTexture(GL_TEXTURE_2D, m_tex_display);

    RTsize bpp = m_output_buffer->getElementSize();
    if(bpp % 8 == 0)
	glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if(bpp % 4 == 0)
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if(bpp % 2 == 0)
	glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    switch(m_output_buffer->getFormat())
    {
    case RT_FORMAT_UNSIGNED_BYTE4:
	glTexSubImage2D(GL_TEXTURE_2D, 0,
		     0,0,m_canvas_size.w, m_canvas_size.h,
		     GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	break;
    case RT_FORMAT_FLOAT4:
	glTexSubImage2D(GL_TEXTURE_2D, 0,
		     0,0,m_canvas_size.w, m_canvas_size.h,
		     GL_RGBA, GL_FLOAT, NULL);
	break;
    case RT_FORMAT_FLOAT3:
	glTexSubImage2D(GL_TEXTURE_2D, 0,
		     0,0,m_canvas_size.w, m_canvas_size.h,
		     GL_RGB, GL_FLOAT, NULL);
	break;
    case RT_FORMAT_FLOAT:
	glTexSubImage2D(GL_TEXTURE_2D, 0,
		     0,0,m_canvas_size.w, m_canvas_size.h,
		     GL_LUMINANCE, GL_FLOAT, NULL);
	break;
    default:
	assert(false && "Unknown buffer format");
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}/*}}}*/
#endif
void canvas::copy_buffer_to_texture()/*{{{*/
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, m_output_buffer->getGLBOId());
    glBindTexture(GL_TEXTURE_2D, m_tex_display);

    RTsize bpp = m_output_buffer->getElementSize();
    if(bpp % 8 == 0)
	glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
    else if(bpp % 4 == 0)
	glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    else if(bpp % 2 == 0)
	glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
    else
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    switch(m_output_buffer->getFormat())
    {
    case RT_FORMAT_UNSIGNED_BYTE4:
	glTexSubImage2D(GL_TEXTURE_2D, 0,
		     0,0,m_canvas_size.w, m_canvas_size.h,
		     GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	break;
    case RT_FORMAT_FLOAT4:
	glTexSubImage2D(GL_TEXTURE_2D, 0,
		     0,0,m_canvas_size.w, m_canvas_size.h,
		     GL_RGBA, GL_FLOAT, NULL);
	break;
    case RT_FORMAT_FLOAT3:
	glTexSubImage2D(GL_TEXTURE_2D, 0,
		     0,0,m_canvas_size.w, m_canvas_size.h,
		     GL_RGB, GL_FLOAT, NULL);
	break;
    case RT_FORMAT_FLOAT:
	glTexSubImage2D(GL_TEXTURE_2D, 0,
		     0,0,m_canvas_size.w, m_canvas_size.h,
		     GL_LUMINANCE, GL_FLOAT, NULL);
	break;
    default:
	assert(false && "Unknown buffer format");
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}/*}}}*/

void canvas::init_gl()/*{{{*/
{
    GLenum err = glewInit();
    if(err != GLEW_OK)
	throw std::runtime_error((char *)glewGetErrorString(err));

    // cria PBO
    GLuint pbo;
    glGenBuffers(1, &pbo);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 1*1*4*sizeof(GLfloat),
		 NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    m_output_buffer = m_ctx->createBufferFromGLBO(RT_BUFFER_OUTPUT, pbo);
    m_output_buffer->setFormat(RT_FORMAT_FLOAT4);
    m_ctx["output_buffer"]->set(m_output_buffer);

    // cria textura 
    glGenTextures(1, &m_tex_display);
    glBindTexture(GL_TEXTURE_2D, m_tex_display);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glBindTexture(GL_TEXTURE_2D, 0);

#if 0

    // cria PBO
    GLuint pbo;
    glGenBuffers(1, &pbo);

    check_glerror();

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    check_glerror();
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 1*1*4*sizeof(GLfloat),
		 NULL, GL_STREAM_DRAW);
    check_glerror();
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    check_glerror();

    m_output_buffer = m_ctx->createBufferFromGLBO(RT_BUFFER_OUTPUT, pbo);
    m_output_buffer->setFormat(RT_FORMAT_FLOAT4);
    m_ctx["output_buffer"]->set(m_output_buffer);

    // cria textura 
    glGenTextures(1, &m_tex_display);
    glBindTexture(GL_TEXTURE_BUFFER, m_tex_display);
    check_glerror();

    glTexParameteri(GL_TEXTURE_BUFFER, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_BUFFER, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    check_glerror();

    glTexBuffer(GL_TEXTURE_BUFFER, GL_RGBA32F, pbo);
    check_glerror();

//    glBindTexture(GL_TEXTURE_BUFFER, 0);

    check_glerror();

#if 0
    glGenFramebuffers(1, &m_fbo);
    check_glerror();
    glBindFramebuffer(GL_READ_FRAMEBUFFER, m_fbo);
    check_glerror();
    glFramebufferTexture(GL_READ_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,
			 m_tex_display,0);
    check_glerror();
    glReadBuffer(GL_COLOR_ATTACHMENT0);
#endif
#endif

    check_glerror();

    // ajusta tamanho das coisas (p/ estado interno ficar OK)
    resize_buffer({1,1});

    m_ctx["root_object"]->set(create_geometry());

    m_ray_buffer = m_ctx->createBuffer(RT_BUFFER_INPUT_OUTPUT |
                                       RT_BUFFER_GPU_LOCAL);
    m_ray_buffer->setFormat(RT_FORMAT_USER);
    m_ray_buffer->setElementSize(sizeof(ray_info));
    m_ray_buffer->setSize(w()*h());
    m_ctx["rays"]->set(m_ray_buffer);

    // configura luzes

    m_lights_buffer = m_ctx->createBuffer(RT_BUFFER_INPUT);
    m_lights_buffer->setFormat(RT_FORMAT_USER);
    m_lights_buffer->setElementSize(sizeof(m_point_lights[0]));
    m_ctx["point_lights"]->set(m_lights_buffer);

#if 0
    shade::point_light light({1,1,1}, 1, {-0.728125,2.98935,-1.59172});
    light.casts_shadow = true;
    m_point_lights.push_back(light);
#endif

#if 1
#if MODEL == SPONZA
    shade::point_light light2({1,1,1}, 1, {0,0,0});
#elif MODEL == TEAPOT
    shade::point_light light2({1,1,1}, 1, {2,2,0});
#endif
    light2.casts_shadow = true;
    m_point_lights.push_back(light2);
#endif

    m_ctx["color"]->setFloat(1,1,1);
    m_ctx["specular_color"]->setFloat(1,1,1);
    m_ctx["kd"]->setFloat(1);
    m_ctx["ka"]->setFloat(0.1);
    m_ctx["ks"]->setFloat(0.6);
    m_ctx["kt"]->setFloat(0);
    m_ctx["shininess"]->setFloat(64);
    m_ctx["ior"]->setFloat(1.51);

    m_ctx["eye"]->setFloat(0,0,0);
    m_ctx["U"]->setFloat(1,0,0);
    m_ctx["V"]->setFloat(0,1,0);
    m_ctx["W"]->setFloat(0,0,1);

    m_ctx->setPrintEnabled(true);

    update_lights();

    m_ctx->validate();
    m_ctx->compile();

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
    glDeleteFramebuffers(1, &m_fbo);

    glDeleteTextures(1, &m_tex_display);
}/*}}}*/

auto create_geom_group(opt::Context &ctx, /*{{{*/
		    const std::initializer_list<opt::GeometryInstance> &objs)
    -> opt::GeometryGroup
{
    opt::GeometryGroup group = ctx->createGeometryGroup();
    group->setChildCount(objs.size());

    int idx=0;
    for(auto it=objs.begin(); it!=objs.end(); ++it, ++idx)
	group->setChild(idx, *it);

    return group;
}/*}}}*/
auto create_group(opt::Context &ctx, /*{{{*/
		    const std::initializer_list<opt::GeometryGroup> &nodes)
    -> opt::Group
{
    opt::Group group = ctx->createGroup();
    group->setChildCount(nodes.size());

    int idx=0;
    for(auto it=nodes.begin(); it!=nodes.end(); ++it, ++idx)
	group->setChild(idx, *it);

    return group;
}/*}}}*/
auto create_sphere(opt::Context &ctx, /*{{{*/
		   const r3::point &center, 
		   float radius, const opt::Material &mat)
    -> opt::GeometryInstance 
{
    opt::Program prog_bounds, prog_isect;
    
    prog_isect = ctx->createProgramFromPTXFile(SPHERE_PTX, "intersect");
    prog_bounds = ctx->createProgramFromPTXFile(SPHERE_PTX, "bounds");

    opt::Geometry sphere = ctx->createGeometry();
    sphere->setPrimitiveCount(1);
    sphere->setBoundingBoxProgram(prog_bounds);
    sphere->setIntersectionProgram(prog_isect);
    sphere["radius"]->setFloat(radius);
    sphere["center"]->setFloat(center.x, center.y, center.z);

    return ctx->createGeometryInstance(sphere, &mat, &mat+1);
}/*}}}*/
auto create_parallelogram(opt::Context &ctx, const r3::point &anchor,/*{{{*/
			  const r3::vector &v1, const r3::vector &v2,
			  const opt::Material &mat)
    -> opt::GeometryInstance 
{
    opt::Program prog_bounds, prog_isect;
    
    prog_isect = ctx->createProgramFromPTXFile(PARALLELOGRAM_PTX,"intersect");
    prog_bounds = ctx->createProgramFromPTXFile(PARALLELOGRAM_PTX, "bounds");

    opt::Geometry parallelogram = ctx->createGeometry();
    parallelogram->setPrimitiveCount(1);
    parallelogram->setBoundingBoxProgram(prog_bounds);
    parallelogram->setIntersectionProgram(prog_isect);

    auto nv1 = v1/dot(v1,v1),
	 nv2 = v2/dot(v2,v2);

    parallelogram["v1"]->setFloat(nv1.x, nv1.y, nv1.z);
    parallelogram["v2"]->setFloat(nv2.x, nv2.y, nv2.z);

    parallelogram["anchor"]->setFloat(anchor.x, anchor.y, anchor.z);

    auto n = unit(cross(v1, v2));
    float d = dot(n, anchor);

    parallelogram["plane"]->setFloat(n.x, n.y, n.z, d);

    return ctx->createGeometryInstance(parallelogram, &mat, &mat+1);
}/*}}}*/
auto create_box(opt::Context &ctx, /*{{{*/
		const r3::point &center, const r3::size &dim,
		const opt::Material &mat)
    -> opt::GeometryInstance
{
    opt::Program prog_bounds, prog_isect;
    
    prog_isect = ctx->createProgramFromPTXFile(BOX_PTX, "intersect");
    prog_bounds = ctx->createProgramFromPTXFile(BOX_PTX, "bounds");

    opt::Geometry box = ctx->createGeometry();
    box->setPrimitiveCount(1);
    box->setBoundingBoxProgram(prog_bounds);
    box->setIntersectionProgram(prog_isect);

    auto boxmin = center-dim/2,
	 boxmax = center+dim/2;

    box["boxmin"]->setFloat(boxmin.x, boxmin.y, boxmin.z);
    box["boxmax"]->setFloat(boxmax.x, boxmax.y, boxmax.z);
    return ctx->createGeometryInstance(box, &mat, &mat+1);
}/*}}}*/
auto create_mesh(opt::Context &ctx, /*{{{*/
		const model &mdl, const r3::point &center,
                const r3::size &size, const opt::Material &mat)
    -> opt::GeometryInstance
{
    opt::Program prog_bounds, prog_isect;
    
    prog_isect = ctx->createProgramFromPTXFile(MESH_PTX, "intersect");
    prog_bounds = ctx->createProgramFromPTXFile(MESH_PTX, "bounds");

    opt::Buffer 
        idx_buffer = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT3,
                                       mdl.faces.size()),
        vtx_buffer = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3,
                                       mdl.positions.size()),
        nor_buffer = ctx->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3,
                                       mdl.normals.size());

    opt::Geometry mesh = ctx->createGeometry();
    mesh->setPrimitiveCount(mdl.faces.size());
    mesh->setBoundingBoxProgram(prog_bounds);
    mesh->setIntersectionProgram(prog_isect);

    mesh["vertex_buffer"]->setBuffer(vtx_buffer);
    mesh["normal_buffer"]->setBuffer(nor_buffer);
    mesh["index_buffer"]->setBuffer(idx_buffer);


    int3 *idx = static_cast<int3 *>(idx_buffer->map());

    for(int f=0; f<mdl.faces.size(); ++f)
        idx[f] = make_int3(mdl.faces[f][0], mdl.faces[f][1], mdl.faces[f][2]);

    idx_buffer->unmap();

    std::vector<r3::point> newpos;
    newpos.reserve(mdl.positions.size());

    r3::box bbox = bounding_box(mdl);

    auto maxdim = max_dim(bbox);

    BOOST_FOREACH(const r3::point &pos, mdl.positions)
        newpos.push_back((pos - (bbox.size/2+bbox.origin))/maxdim*size+center);

    memcpy(vtx_buffer->map(), &newpos[0], sizeof(float)*newpos.size()*3);
    vtx_buffer->unmap();

    memcpy(nor_buffer->map(), &mdl.normals[0], 
           sizeof(float)*mdl.normals.size()*3);
    nor_buffer->unmap();

    return ctx->createGeometryInstance(mesh, &mat, &mat+1);

}/*}}}*/
auto create_flat_shade(opt::Context &ctx)/*{{{*/
    -> opt::Material
{
    opt::Material mat = ctx->createMaterial();
    opt::Program prog_hit;
    prog_hit = ctx->createProgramFromPTXFile(FLAT_MATERIAL_PTX,"closest_hit");
    mat->setClosestHitProgram(0, prog_hit);

    return mat;
}/*}}}*/
auto create_plastic_material(opt::Context &ctx)/*{{{*/
    -> opt::Material
{
    opt::Material mat = ctx->createMaterial();
    opt::Program prog_hit;
    prog_hit = ctx->createProgramFromPTXFile(PLASTIC_MATERIAL_PTX,"closest_hit");
    mat->setClosestHitProgram(0, prog_hit);

    opt::Program any_hit;
    any_hit = ctx->createProgramFromPTXFile(PLASTIC_MATERIAL_PTX,"any_hit_shadow");
    mat->setAnyHitProgram(1, any_hit);

    return mat;
}/*}}}*/
auto create_matte_material(opt::Context &ctx)/*{{{*/
    -> opt::Material
{
    opt::Material mat = ctx->createMaterial();
    opt::Program prog_hit;
    prog_hit = ctx->createProgramFromPTXFile(MATTE_MATERIAL_PTX,"closest_hit");
    mat->setClosestHitProgram(0, prog_hit);

    opt::Program any_hit;
    any_hit = ctx->createProgramFromPTXFile(PLASTIC_MATERIAL_PTX,"any_hit_shadow");
    mat->setAnyHitProgram(1, any_hit);

    return mat;
}/*}}}*/

#if 0
opt::GeometryGroup canvas::create_geometry()/*{{{*/
{
    auto plastic = create_plastic_material(m_ctx);

    auto sphere = create_sphere(m_ctx, {0.5,0,0.5}, 0.25, plastic);
    sphere["color"]->setFloat(1,1,0); // yellow
    sphere["ks"]->setFloat(0.7);

    auto box = create_box(m_ctx, {-0.5,0,0.5},{0.5,0.5,0.5}, plastic);
    box["color"]->setFloat(0,0,1); // blue
    box["ks"]->setFloat(0);

    auto ground = create_parallelogram(m_ctx, {1,-0.25,1},{-2,0,0},{0,0,-2},
				       plastic);
    ground["color"]->setFloat(.3,.3,.3); // grey
    ground["ks"]->setFloat(0.5);

    auto base_group = create_geom_group(m_ctx, {sphere, box, ground});
    base_group->setAcceleration(m_ctx->createAcceleration("Sbvh", "Bvh"));

    auto teapot = create_mesh(m_ctx, load_model("teapot.ply"), 
                              {0,0,-0.5}, {1,1,1}, plastic);
    teapot["color"]->setFloat(0,1,0); // green

#if 0
    auto group_teapot = create_geom_group(m_ctx,{teapot});

    auto accel_teapot = m_ctx->createAcceleration("Sbvh", "Bvh");
    accel_teapot->setProperty("vertex_buffer_name","vertex_buffer");
    accel_teapot->setProperty("index_buffer_name","index_buffer");

    group_teapot->setAcceleration(accel_teapot);
#endif
    auto group = create_geom_group(m_ctx, {sphere,box,ground,teapot});

//    auto group = create_group(m_ctx, {base_group, group_teapot});

    group->setAcceleration(m_ctx->createAcceleration("Sbvh", "Bvh"));
    //group->setAcceleration(m_ctx->createAcceleration("NoAccel", "NoAccel"));
    
    return group;
}/*}}}*/
#endif
#if MODEL == SPONZA
opt::GeometryGroup canvas::create_geometry()/*{{{*/
{
    auto plastic = create_matte_material(m_ctx);

    auto sponza = create_mesh(m_ctx, load_model("sponza.ply"), 
                              {0,0,0}, {1,1,1}, plastic);
    sponza["color"]->setFloat(1,1,1); // green

    auto group = create_geom_group(m_ctx, {sponza});

    group->setAcceleration(m_ctx->createAcceleration("Sbvh", "Bvh"));
    
    return group;
}/*}}}*/
#endif
#if MODEL == TEAPOT
opt::GeometryGroup canvas::create_geometry()/*{{{*/
{
    auto plastic = create_matte_material(m_ctx);

    auto teapot = create_mesh(m_ctx, load_model("teapot.ply"), 
                              {0,0,0}, {2,2,2}, plastic);
//    teapot["color"]->setFloat(1,1,1); // green

#if 0
    auto group_teapot = create_geom_group(m_ctx,{teapot});

    auto accel_teapot = m_ctx->createAcceleration("Sbvh", "Bvh");
    accel_teapot->setProperty("vertex_buffer_name","vertex_buffer");
    accel_teapot->setProperty("index_buffer_name","index_buffer");

    group_teapot->setAcceleration(accel_teapot);
#endif
    auto group = create_geom_group(m_ctx, {teapot});

//    auto group = create_group(m_ctx, {base_group, group_teapot});

    group->setAcceleration(m_ctx->createAcceleration("Sbvh", "Bvh"));
    //group->setAcceleration(m_ctx->createAcceleration("NoAccel", "NoAccel"));
    
    return group;
}/*}}}*/
#endif
void canvas::update_lights()/*{{{*/
{
    RTsize bufsize;
    m_lights_buffer->getSize(bufsize);

    if(m_point_lights.size() != bufsize)
	m_lights_buffer->setSize(m_point_lights.size());

    memcpy(m_lights_buffer->map(), &m_point_lights[0],
	   m_point_lights.size()*sizeof(m_point_lights[0]));
    m_lights_buffer->unmap();
}/*}}}*/

void canvas::raytrace()
{
#if 1
#if MODEL == SPONZA
    r3::point light_pos(0,0,0);
#elif MODEL == TEAPOT
    r3::point light_pos(2,2,0);
#endif

    auto light_rot = m_light_drag_rot*m_light_rot;

    light_pos = rot(light_pos, light_rot);

//    light_pos = {-0.728125,2.98935,-1.59172};

    m_point_lights[0].pos = {light_pos.x, light_pos.y, light_pos.z};
    update_lights();
//    std::cout << "LIGHT: " << light_pos << std::endl;
#endif


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

    m_ctx["U"]->setFloat(u.x, u.y, u.z);
    m_ctx["V"]->setFloat(v.x, v.y, v.z);
    m_ctx["W"]->setFloat(w.x, w.y, w.z);
    m_ctx["eye"]->setFloat(frustum.pos.x, frustum.pos.y, frustum.pos.z);

    m_ctx->launch(0, m_canvas_size.w,m_canvas_size.h);
}

void canvas::draw()
{
    if(!m_glOK)
	init_gl();

    if(m_canvas_size != r2::isize(w(), h()))
	resize_buffer({w(), h()});

    raytrace();
    copy_buffer_to_texture();

#if 0
    glBlitFramebuffer(0,0,m_canvas_size.w-1, m_canvas_size.h-1,
		      0,0,m_canvas_size.w-1, m_canvas_size.h-1,
		      GL_COLOR_BUFFER_BIT, GL_NEAREST);
#endif

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, m_tex_display);

    // pixel center sampling
    float pu = 0.5/m_canvas_size.w,
	  pv = 0.5/m_canvas_size.h;

    glBegin(GL_QUADS);
	glTexCoord2f(pu,pv);
	glVertex2f(-1,-1);

	glTexCoord2f(1,pv);
	glVertex2f(1,-1);

	glTexCoord2f(1-pu,1-pv);
	glVertex2f(1,1);

	glTexCoord2f(pu,1-pv);
	glVertex2f(-1,1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
}
