#ifndef S3D_MPL_EQUAL_H
#define S3D_MPL_EQUAL_H

#include <type_traits>
#include "vector_fwd.h"
#include "bool_fwd.h"
#include "unref.h"


namespace s3d { namespace mpl
{

	namespace detail
	{
		template <class V1, class V2> 
		struct equal_impl
			: equal_impl<typename unref<V1>::type, typename unref<V2>::type>
		{
		};

		template <class T> struct equal_impl<vector_c<T>, vector_c<T>>
		{
			static const bool value = true;
		};

		template <class T, T I, T...II, T J, T...JJ> 
		struct equal_impl<vector_c<T,I,II...>, vector_c<T,J,JJ...>>
		{
		private:
			typedef typename std::conditional
			<
				I == J,
				equal_impl<vector_c<T,II...>, vector_c<T,JJ...>>,
				bool_<false>
			>::type typeval;

		public:
			static const bool value = typeval::value;
		};

		template <> struct equal_impl<vector<>, vector<>>
		{
			static const bool value = true;
		};

		template <class T, class...TT, class U, class...UU> 
		struct equal_impl<vector<T, TT...>, vector<U, UU...>>
		{
		private:
			typedef typename std::conditional
			<
				std::is_same<T,U>::value,
				equal_impl<vector<TT...>, vector<UU...>>,
				bool_<false>
			>::type typeval;

		public:
			static const bool value = typeval::value;
		};
	}

	template <class V1, class V2> 
	struct equal
	{
	private:
		typedef typename std::conditional
		<
			unref<V1>::type::size == unref<V2>::type::size,
			detail::equal_impl<V1,V2>,
			bool_<false>
		>::type typeval;

	public:
		static const bool value = typeval::value;
	};


}} // namespace s3d::mpl

#endif
