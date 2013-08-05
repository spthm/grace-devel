#ifndef S3D_MATH_DETAIL_H
#define S3D_MATH_DETAIL_H

namespace s3d { namespace math { namespace detail
{

template <class T, class U> struct equaldim
	: equaldim<Vector<float, T::dim>, Vector<float, U::dim>>
{
};

template <class T, class U> 
struct equaldim<T &, U> : equaldim<typename std::remove_cv<T>::type, U> {};

template <class T, class U> 
struct equaldim<T, U&> : equaldim<T, typename std::remove_cv<U>::type> {};

template <class T, class U> 
struct equaldim<T&, U&> : equaldim<typename std::remove_cv<T>::type, 
									typename std::remove_cv<U>::type> {};

template <template<class> class V1, template<class> class V2,
		 class T1, class T2>
struct equaldim<V1<T1>,V2<T2>>
{
	static const bool value = true;
};

#if GCC_VERSION < 40500 || true
template <template<class,int...> class V1, template<class,int...> class V2,
		  class T1, class T2, int N, int P>
struct equaldim<V1<T1,N>,V2<T2,P>>
{
	static const bool value = (N==RUNTIME || P==RUNTIME || N==P);
};
#endif

template <template<class,int...> class V1, template<class,int...> class V2,
		  class T1, class T2, int N, int P, int... D1, int... D2>
struct equaldim<V1<T1,N,D1...>,V2<T2,P,D2...>>
{
	static_assert(sizeof...(D1) == sizeof...(D2), 
				  "Dimension count must be equal");

	static const bool value = (N==RUNTIME || P==RUNTIME || N==P) &&
							  equaldim<V1<T1,D1...>,V2<T2,D2...>>::value;
};

}}} // namespace s3d::math::detail

#endif
