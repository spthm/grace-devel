#ifndef S3D_MATH_VECTOR_VIEW_H
#define S3D_MATH_VECTOR_VIEW_H

#include "coords.h"
#include "euclidean_space.h"

namespace s3d { namespace math
{

template <class T, int N> 
struct VectorView 
	: coords<VectorView<T,N>,euclidean_space_view<T,N>>
{
private:
	typedef coords<VectorView,euclidean_space_view<T,N>> coords_base;

public:
	template <class U, int D=N> struct rebind {typedef VectorView<U,D> type;};

	VectorView(T *data, size_t size) /*{{{*/
		: coords_base(data, size)
	{
	}/*}}}*/

	template <class DUMMY=int,
		class = typename std::enable_if<sizeof(DUMMY)!=0 && N!=RUNTIME>::type> 
	VectorView(T *data) /*{{{*/
		: coords_base(data)
	{
	}/*}}}*/

	VectorView(const VectorView &p) = default;

	template <class V, 
		class = typename std::enable_if<is_vector<V>::value>::type>
	VectorView &operator =(const V &that);

	using coords_base::size;
	std::size_t cols() const { return size(); }
};

template <class T, int D>
struct is_vector<VectorView<T,D>>
{
	static const bool value = true;
};


}} // namespace s3d::math

#include "vector_view.hpp"

#endif
