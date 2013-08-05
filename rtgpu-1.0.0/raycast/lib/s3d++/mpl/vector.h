#ifndef S3D_MPL_VECTOR_H
#define S3D_MPL_VECTOR_H

namespace s3d { namespace mpl
{
	template <class...ARGS> struct vector
	{
		typedef vector<ARGS...> type;
		static const std::size_t size = sizeof...(ARGS);
	};

	template <class T, T...II> struct vector_c
	{
		typedef T value_type;
		typedef vector_c<T,II...> type;

		static const std::size_t size = sizeof...(II);
	};
}} // namespace s3d::mpl

#endif
