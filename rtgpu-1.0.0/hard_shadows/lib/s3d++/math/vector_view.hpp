#include "detail.h"

namespace s3d { namespace math
{

template <class T, int N> template <class V, class>
auto VectorView<T,N>::operator =(const V &that) -> VectorView &
{
	static_assert(detail::equaldim<VectorView, V>::value, "Mismatched dimensions");

	// Assignment works if this is an empty vector (needed for swap)
	if(!this->empty() && size() != that.size())
		throw std::runtime_error("Mismatched dimensions");

	coords_base::operator=(that);
	return *this;
}

template <class T, int N>
void swap(VectorView<T,N> &a, VectorView<T,N> &b)
{
	swap_ranges(a.begin(), a.end(), b.begin());
}


}} // namespace s3d::math
