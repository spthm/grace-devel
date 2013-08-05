#include "pch.h"
#include "model.h"

using namespace s3d;

r3::box bounding_box(const model &m)/*{{{*/
{
    r3::box bbox = r3::null_box;
    for(auto it=m.positions.begin(); it!=m.positions.end(); ++it)
	bbox.merge(*it);

    return bbox;
}/*}}}*/

namespace/*{{{*/
{
    template <class T, typename T::value_type T::space_type::* M>
	int setv(const p_ply_argument arg)
	{
	    void *pdata;
	    ply_get_argument_user_data(arg, &pdata, NULL);
	    auto &v = *reinterpret_cast<std::vector<T> *>(pdata);

	    long index;
	    ply_get_argument_element(arg, NULL, &index);

	    assert(index < v.size());

	    v[index].*M = ply_get_argument_value(arg);
	    return 1;
	}

    int setf(const p_ply_argument arg)
    {
	void *pdata;
	ply_get_argument_user_data(arg, &pdata, NULL);
	auto &faces = *reinterpret_cast<std::vector<math::face<int,3>> *>(pdata);

	long length, value_index;
	ply_get_argument_property(arg, NULL, &length, &value_index);

	// only triangles this time
	if(length != 3)
	    return 0;

	if(value_index == -1)
	    faces.emplace_back(-1,-1,-1);
	else
	{
	    long instance_index;
	    ply_get_argument_element(arg, NULL, &instance_index);

	    faces.back()[value_index] = ply_get_argument_value(arg);
	}

	return 1;
    }
}/*}}}*/

model load_model_ply(const std::string &fname)/*{{{*/
{
    auto ply = ply_open(fname.c_str(), NULL, 0, NULL);
    if(!ply)
	throw std::runtime_error("Unable to open "+fname);

    try
    {
	if(!ply_read_header(ply))
	    throw std::runtime_error("Not a valid ply file");

	model m;


	if(int cnt = ply_set_read_cb(ply,"vertex","x",
				     &setv<r3::point, &r3::point::x>, 
				     &m.positions, 0))
	{
	    m.positions.resize(cnt);

	    ply_set_read_cb(ply, "vertex", "y", 
			    &setv<r3::point,&r3::point::y>, 
			    &m.positions, 0);

	    ply_set_read_cb(ply, "vertex", "z", 
			    &setv<r3::point, &r3::point::z>, 
			    &m.positions, 0);
	}
	else
	    throw std::runtime_error("Empty mesh (no vertices)");

#if 1
	if(int cnt = ply_set_read_cb(ply,"vertex","nx",
				     &setv<r3::vector, &r3::vector::x>, 
				     &m.normals, 0))
	{
	    m.normals.resize(cnt);

	    ply_set_read_cb(ply, "vertex", "ny", 
			    &setv<r3::vector,&r3::vector::y>, 
			    &m.normals, 0);

	    ply_set_read_cb(ply, "vertex", "nz", 
			    &setv<r3::vector,&r3::vector::z>,
			    &m.normals, 0);
	}
#endif

	if(int cnt = ply_set_read_cb(ply,"vertex","red",
				     &setv<color::rgb, &color::rgb::r>, 
				     &m.colors, 0))
	{
	    m.colors.resize(cnt);

	    ply_set_read_cb(ply, "vertex", "green", 
			    &setv<color::rgb, &color::rgb::g>, 
			    &m.colors, 0);

	    ply_set_read_cb(ply, "vertex", "blue", 
			    &setv<color::rgb, &color::rgb::b>, 
			    &m.colors, 0);
	}
	else if(int cnt = ply_set_read_cb(ply,"vertex","diffuse_red",
					  &setv<color::rgb, &color::rgb::r>, 
					  &m.colors, 0))
	{
	    m.colors.resize(cnt);

	    ply_set_read_cb(ply, "vertex", "diffuse_green", 
			    &setv<color::rgb, &color::rgb::g>, 
			    &m.colors, 0);

	    ply_set_read_cb(ply, "vertex", "diffuse_blue", 
			    &setv<color::rgb, &color::rgb::b>, 
			    &m.colors, 0);
	}

	if(int cnt = ply_set_read_cb(ply,"vertex","tx",
				     &setv<r2::param_coord, &r2::param_coord::u>, 
				     &m.texcoords, 0))
	{
	    m.texcoords.resize(cnt);
	    ply_set_read_cb(ply, "vertex", "ty", 
			    &setv<r2::param_coord,&r2::param_coord::v>, 
			    &m.texcoords, 0);
	}
	else if(int cnt = ply_set_read_cb(ply,"vertex","s",
					  &setv<r2::param_coord, &r2::param_coord::u>,
					  &m.texcoords, 0))
	{
	    m.texcoords.resize(cnt);
	    ply_set_read_cb(ply, "vertex", "t", 
			    &setv<r2::param_coord,&r2::param_coord::v>, 
			    &m.texcoords, 0);
	}

	int cnt = ply_set_read_cb(ply, "face", "vertex_indices", 
				  &setf, &m.faces, 0);
	if(cnt == 0)
	    throw std::runtime_error("Empty mesh (no faces)");

	if(!ply_read(ply))
	    throw std::runtime_error("Error reading ply");

	if(m.normals.empty())
	{
	    m.normals.resize(m.positions.size(), r3::vector(0,0,0));

	    for(int i=0; i<m.faces.size(); ++i)
	    {
		auto pos = make_face_view(m.faces[i], m.positions);

		r3::vector n = cross(pos[1]-pos[0], pos[2]-pos[0]);

		if(sqrnorm(n) == 0)
		    continue;

		auto nor = make_face_view(m.faces[i], m.normals);

		for(int j=0; j<3; ++j)
		    nor[j] += n;
	    }

	    for(int i=0; i<m.normals.size(); ++i)
	    {
		if(sqrnorm(m.normals[i]) > 0)
		    m.normals[i] = unit(m.normals[i]);
	    }
	}
	return std::move(m);
    }
    catch(...)
    {
	ply_close(ply);
	throw;
    }
}/*}}}*/

model load_model_smf(const std::string &fname)/*{{{*/
{
    std::ifstream in(fname);

    if(!in)
	throw std::runtime_error("Error opening "+fname);

    std::string str;
    in >> str;

    if(str != "begin")
	throw std::runtime_error("Bad msf format");

    model mdl;

    while(in)
    {
	char c;
	in >> c;
	switch(c)
	{
	case 'v':
	    mdl.positions.emplace_back();
	    in >> mdl.positions.back();
	    break;
	case 'f':
	    {
		r3::point face;

		in >> face;
		mdl.faces.emplace_back(face[0]-1, face[1]-1, face[2]-1);

		auto pos = make_face_view(mdl.faces.back(), mdl.positions);

		r3::vector n = cross(pos[1]-pos[0], pos[2]-pos[0]);

		for(int i=0; i<3; ++i)
		{
		    // this is not fast, I know
		    mdl.normals.resize(max(mdl.normals.size(),face[i]), 
				       r3::vector(0,0,0));
		    mdl.normals[face[i]-1] += n;
		}
	    }
	    break;
	case 'e':
	    in >> str;
	    if(str == "nd")
		break;
	    // fall-through
	default:
	    throw std::runtime_error("Bad msf format"); // I could do better...
	}
    }

    for(int i=0; i<mdl.normals.size(); ++i)
    {
	if(sqrnorm(mdl.normals[i]) > 0)
	    mdl.normals[i] = unit(mdl.normals[i]);
    }

    return std::move(mdl);
}
/*}}}*/

model load_model(const std::string &fname)
{
    if(fname.find(".ply") != fname.npos)
	return load_model_ply(fname);
    else if(fname.find(".smf") != fname.npos)
	return load_model_smf(fname);
    else
	throw std::runtime_error("I cannot handle this mesh");
}
