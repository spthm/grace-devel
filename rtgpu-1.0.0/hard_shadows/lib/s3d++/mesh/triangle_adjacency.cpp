#include "triangle_adjacency.h"
#include "../math/r2/box.h"
#include "../math/r2/param_coord.h"
#include "surface.h"

namespace s3d { namespace math
{

auto make_triangle_adjacency_strip_grid(std::vector<r2::param_coord> &coords,
										  const r2::box &a, size_t nu,size_t nv)
	-> surface<triangle_adjacency_strip<int>>
{
	surface<triangle_adjacency_strip<int>> surf;

	if(nu==0 || nv==0)
		return surf;

	real u = a.x,
		 v = a.y+a.h,
		 du = a.w/nu,
		 dv = -a.h/nv;

	coords.reserve(coords.size()+(nu+3)*(nv+3));
	surf.reserve(nv);

	triangle_adjacency_strip<int> strip;

	// first strip, left to right
	strip.reserve((nu+1)*4*nv);

	coords.emplace_back(u,v);
	strip.push_back(coords.size()-1);

	coords.emplace_back(u-du,v+dv);
	strip.push_back(coords.size()-1);

	for(size_t i=0; i<nu; ++i, u += du)
	{
		coords.emplace_back(u, v+dv);
		strip.push_back(coords.size()-1);

		coords.emplace_back(u+du, v-dv);
		strip.push_back(coords.size()-1);

		coords.emplace_back(u+du, v);
		strip.push_back(coords.size()-1);

		coords.emplace_back(u, v+2*dv);
		strip.push_back(coords.size()-1);
	}

	coords.emplace_back(u,v+dv);
	strip.push_back(coords.size()-1);

	coords.emplace_back(u+du,v);
	strip.push_back(coords.size()-1);

	surf.push_back(std::move(strip));
	strip.clear();

	v += dv;

	// middle strips
	for(size_t i=1; i<nv; ++i, v += dv)
	{
		u = a.x;

		auto &prev_strip = surf.back();

		strip.push_back(prev_strip[2]);

		coords.emplace_back(u-du,v+dv);
		strip.push_back(coords.size()-1);

		for(size_t j=0; j<nu; ++j, u += du)
		{
			int idx = strip.size();
			strip.push_back(prev_strip[idx+3]);
			strip.push_back(prev_strip[idx+2]);
			strip.push_back(prev_strip[idx+4]);

			coords.emplace_back(u, v+2*dv);
			strip.push_back(coords.size()-1);
		}

		coords.emplace_back(u,v+dv);
		strip.push_back(coords.size()-1);

		coords.emplace_back(u+du,v);
		strip.push_back(coords.size()-1);

		surf.push_back(std::move(strip));
		strip.clear();
	}

	return std::move(surf);
}

}} // namespace s3d::mesh
