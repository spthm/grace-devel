#ifndef S3D_UTIL_FUNCTOR_H
#define S3D_UTIL_FUNCTOR_H

namespace s3d
{

	/*
namespace detail
{
	template <class T> struct construct_impl
	{
		template <class...ARGS>
		T operator()(ARGS &&...args) const
		{
			return T(std::forward<ARGS>(args)...);
		}
	};
}

template <class T, class...ARGS>
detail::construct_impl<T> construct(ARGS &&...args)
{
	return detail::construct_impl<T>(args...);
}
*/


namespace detail
{
	template <class T> struct var_impl
	{
	public:
		var_impl(T v) : m_var(std::forward<T>(v)) {}

		template <class...ARGS>
		T operator()(ARGS &&...args) const
		{
			return m_var;
		}
	private:
		T m_var;
	};
}

template <class T>
detail::var_impl<T> var(T var)
{
	return detail::var_impl<T>(std::forward<T>(var));
}


struct null_functor
{
	template <class...ARGS>
	null_functor(ARGS &...) {};
};

}

#endif
