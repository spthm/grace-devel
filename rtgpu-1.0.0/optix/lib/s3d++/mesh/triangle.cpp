#include "surface.h"
#include "../math/r2/param_coord.h"
#include "../math/r2/box.h"

namespace s3d { namespace math
{

/* creates a rectangular mesh composed by CCW triangle strips

   02468A
   13579B
   HGFEDC
   IJKLMN
   TSRQPO
*/

auto make_triangle_grid(std::vector<r2::param_coord> &coords,
				 const r2::box &a, size_t nu, size_t nv)
	-> surface<face<int,3>>
{
	surface<face<int,3>> surf;

	if(nu==0 || nv==0)
		return surf;

	real u = a.x,
		 v = a.y+a.h,
		 du = a.w/nu,
		 dv = -a.h/nv;

	coords.reserve(coords.size()+(nu+1)*(nv+1));
	surf.reserve(nu*nv*2);

	// first triangle: top-left
	// coord origin: bottom-left

	coords.emplace_back(u, v);
	coords.emplace_back(u, v+=dv);

	for(size_t j=0; j<nu; ++j)
	{
		u += du;
		coords.emplace_back(u, v-dv);

		int idx = coords.size()-1;
		surf.make_face(idx-2, idx-1, idx); 

		coords.emplace_back(u, v);

		surf.make_face(idx, idx-1, idx+1); 
	}

	if(nv == 1)
		return std::move(surf);


	// second row (right to left)
	v += dv;
	coords.emplace_back(u, v);

	int back = 2;

	for(size_t j=0; j<nu; ++j, back+=3)
	{
		coords.emplace_back(u-=du, v);

		int idx = coords.size()-1;

		surf.make_face(idx-1, idx-back, idx); 
		surf.make_face(idx, idx-back, idx-back-2); 
	}

	if(nv == 2)
		return std::move(surf);


	for(size_t i=2; i<nv; i+=2)
	{
		// left to right
		v += dv;
		coords.emplace_back(u, v);

		back = 3;

		for(size_t j=0; j<nu; ++j, back+=2)
		{
			coords.emplace_back(u+=du, v);
			int idx = coords.size()-1;

			surf.make_face(idx-back+1, idx-1, idx-back); 
			surf.make_face(idx-back, idx-1, idx); 

		}

		if(i == nv-1)
			break;
	
		// right to left
		v += dv;
		coords.emplace_back(u, v);

		back = 3;

		for(size_t j=0; j<nu; ++j, back+=2)
		{
			coords.emplace_back(u-=du, v);
			int idx = coords.size()-1;

			surf.make_face(idx-1, idx-back+1, idx); 
			surf.make_face(idx, idx-back+1, idx-back); 
		}
	}

	return std::move(surf);
}

}} // namespace s3d::s3d

// $Id: util.cpp 2995 2010-08-23 00:40:45Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

