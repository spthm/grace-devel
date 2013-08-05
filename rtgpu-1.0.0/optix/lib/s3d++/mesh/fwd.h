#ifndef S3D_MESH_FWD_H
#define S3D_MESH_FWD_H

#include "../math/fwd.h"

namespace s3d { namespace math
{

template <class V, int D=RUNTIME> class face;
template <class F> class surface;
template <class S> class mesh;

template <class T, int D=RUNTIME> class face_view;
template <class T> class surface_view;
template <class T> class mesh_view;

// should be a template alias to face<T,3>
//template <class T> class triangle;
template <class T> class triangle_strip;
template <class T> class triangle_adjacency;
template <class T> class triangle_adjacency_strip;

template <class T> struct is_face;

}

using math::face;
using math::surface;
using math::mesh;
using math::face_view;
using math::surface_view;
using math::mesh_view;
using math::triangle_strip;
using math::triangle_adjacency;
using math::triangle_adjacency_strip;

} // namespace s3d::math

#endif
