#ifndef S3D_UTIL_MULT_PARAM_H
#define S3D_UTIL_MULT_PARAM_H

#include <boost/functional/hash.hpp>

namespace s3d
{

template <class T, int D, class TAG=void>
class multi_param
{
public:
	typedef std::array<T, D> container_type;
	typedef T value_type;
	static const std::size_t dim = D;

	typedef typename container_type::iterator 
		iterator;
	typedef typename container_type::const_iterator 
		const_iterator;

	typedef typename container_type::reverse_iterator 
		reverse_iterator;
	typedef typename container_type::const_reverse_iterator 
		const_reverse_iterator;

	template<class...ARGS, class =
		typename std::enable_if<sizeof...(ARGS)+1==D>::type>
	multi_param(T d, ARGS... dims) 
		: m_params((container_type){{d, dims...}}) {};

	multi_param() {} // TODO: remove this someday

	const T &operator[](int i) const 
		{ assert(i >= 0 && i < D); return m_params[i]; }

	T &operator[](int i)
		{ assert(i >= 0 && i < D); return m_params[i]; }

	bool operator==(const multi_param &that) const
	{
		return std::equal(m_params.begin(),m_params.end(),that.m_params.begin());
	}

	bool operator!=(const multi_param &that) const
		{ return !operator==(that); }

	iterator begin() { return m_params.begin(); }
	iterator end() { return m_params.end(); }

	const_iterator begin() const { return m_params.begin(); }
	const_iterator end() const { return m_params.end(); }

	reverse_iterator rbegin() { return m_params.rbegin(); }
	reverse_iterator rend() { return m_params.rend(); }

	const_reverse_iterator rbegin() const { return m_params.rbegin(); }
	const_reverse_iterator rend() const { return m_params.rend(); }

	const_iterator cbegin() { return m_params.cbegin(); }
	const_iterator cend() { return m_params.cend(); }

	const_iterator cbegin() const { return m_params.cbegin(); }
	const_iterator cend() const { return m_params.cend(); }

	const_reverse_iterator crbegin() { return m_params.rbegin(); }
	const_reverse_iterator crend() { return m_params.rend(); }

	const_reverse_iterator crbegin() const { return m_params.crbegin(); }
	const_reverse_iterator crend() const { return m_params.crend(); }

	friend size_t hash_value(const multi_param &p)
	{
		return boost::hash_range(p.m_params.begin(), p.m_params.end());
	}

private:
	container_type m_params;
};

} // namespace s3d

namespace std
{
	template <class T, int D, class TAG>
	struct hash<s3d::multi_param<T,D,TAG>>
	{
		size_t operator()(const s3d::multi_param<T,D,TAG> &d) const
		{
			return hash_value(d);
		}
	};
}


#endif
