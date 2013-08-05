#ifndef S3D_MATH_EUCLIDEAN_SPACE_H
#define S3D_MATH_EUCLIDEAN_SPACE_H

#include "../util/memory_view.h"

namespace s3d { namespace math
{

namespace detail
{
	template <class T>
	T &null_value()
	{
		static T val;
		return val;
	}
}

template <class T, int D> class euclidean_space_view/*{{{*/
{
	typedef memory_view<T,D> container_type;
public:
	typedef T value_type;
	static const int dim = D;

	euclidean_space_view(T *data)
		: m_coords(data) {}

	euclidean_space_view &operator=(const euclidean_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	euclidean_space_view &operator=(const euclidean_space_view<U,D> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

	container_type m_coords;
};/*}}}*/
template <class T> class euclidean_space_view<T,1>/*{{{*/
{
public:
	static const int dim = 1;
	typedef T value_type;

	euclidean_space_view(const euclidean_space_view &that) 
		: x(that.x) {}

	euclidean_space_view(T *data)
		: x(data ? data[0] : detail::null_value<T>())
	{
	}

	euclidean_space_view &operator=(const euclidean_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	euclidean_space_view &operator=(const euclidean_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &x; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &x; }
	const_iterator end() const { return begin()+dim; }

	T &x;
};/*}}}*/
template <class T> class euclidean_space_view<T,2>/*{{{*/
{
public:
	static const int dim = 2;
	typedef T value_type;

	euclidean_space_view(const euclidean_space_view &that) 
		: x(that.x), y(that.y) {}
	euclidean_space_view(T *data)
		: x(data ? data[0] : detail::null_value<T>())
		, y(data ? data[1] : detail::null_value<T>())
	{
		// data might be null
	}

	euclidean_space_view &operator=(const euclidean_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	euclidean_space_view &operator=(const euclidean_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &x; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &x; }
	const_iterator end() const { return begin()+dim; }

	T &x, &y;
};/*}}}*/
template <class T> class euclidean_space_view<T,3>/*{{{*/
{
public:
	static const int dim = 3;
	typedef T value_type;

	euclidean_space_view(const euclidean_space_view &that) 
		: x(that.x), y(that.y), z(that.z) {}

	euclidean_space_view(T *data)
		: x(data ? data[0] : detail::null_value<T>())
		, y(data ? data[1] : detail::null_value<T>())
		, z(data ? data[2] : detail::null_value<T>())
	{
		// data might be null
	}

	euclidean_space_view &operator=(const euclidean_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	euclidean_space_view &operator=(const euclidean_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &x; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &x; }
	const_iterator end() const { return begin()+dim; }

	T &x, &y, &z;
};/*}}}*/
template <class T> class euclidean_space_view<T,4>/*{{{*/
{
	typedef memory_view<T,4> container_type;
public:
	static const int dim = 4;
	typedef T value_type;

	euclidean_space_view(const euclidean_space_view &that) 
		: x(that.x), y(that.y), z(that.z), w(that.w) {}

	euclidean_space_view(T *data)
		: x(data ? data[0] : detail::null_value<T>())
		, y(data ? data[1] : detail::null_value<T>())
		, z(data ? data[2] : detail::null_value<T>())
		, w(data ? data[3] : detail::null_value<T>())
	{
		assert(data);
	}

	euclidean_space_view &operator=(const euclidean_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	euclidean_space_view &operator=(const euclidean_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &x; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &x; }
	const_iterator end() const { return begin()+dim; }

	T &x, &y, &z, &w;
};/*}}}*/
template <class T> class euclidean_space_view<T,RUNTIME>/*{{{*/
{
	typedef memory_view<T> container_type;
public:
	static const int dim = RUNTIME;
	typedef T value_type;

	euclidean_space_view(const euclidean_space_view &that) 
		: m_coords(that.m_coords) {}

	euclidean_space_view(T *data, size_t size)
		: m_coords(data, size)
	{
		assert(size==0 || data!=NULL);
	}

	euclidean_space_view &operator=(const euclidean_space_view &that)
	{
		if(m_coords.size() != that.m_coords.size())
			throw std::runtime_error("Mismatched dimension");

		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	euclidean_space_view &operator=(const euclidean_space_view<U,dim> &that)
	{
		if(m_coords.size() != that.m_coords.size())
			throw std::runtime_error("Mismatched dimension");

		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

	void reassign(T *data, size_t size)
	{
		m_coords = {data, size};
	}

	memory_view<T> m_coords;
};/*}}}*/

template <class T, int N=RUNTIME> class euclidean_space/*{{{*/
{
	typedef std::array<typename std::remove_const<T>::type,N> container_type;
public:
	static_assert(N > 0, "Invalid dimension");

	static const int dim = N;
	typedef T value_type;

	euclidean_space() {}

	template <class U>
	euclidean_space(const euclidean_space<U,N> &that) 
		: m_coords(that.m_coords) {}

	euclidean_space(const dimension<1>::type &d)
	{
		if(d[0] != N)
			throw std::runtime_error("Mismatched dimension");
	}
	template <class...ARGS>
	euclidean_space(const value_type &v1, ARGS &&...args)
		: m_coords((container_type){{v1, value_type(args)...}}) {}

	template <class U>
	euclidean_space &operator=(const euclidean_space<U,N> &that) 
	{
		m_coords = that.m_coords;
		return *this;
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

private:
	container_type m_coords;
};/*}}}*/
template <class T> class euclidean_space<T,1>/*{{{*/
{
	typedef std::array<typename std::remove_const<T>::type,1> container_type;
public:
	static const int dim = 1;
	typedef T value_type;

	template <class U>
	euclidean_space &operator=(const euclidean_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	euclidean_space &operator=(const euclidean_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &x; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &x; }
	const_iterator end() const { return begin()+dim; }

	T x;
};/*}}}*/
template <class T> class euclidean_space<T,2>/*{{{*/
{
	typedef std::array<typename std::remove_const<T>::type,2> container_type;
public:
	static const int dim = 2;
	typedef T value_type;

	template <class U>
	euclidean_space &operator=(const euclidean_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	euclidean_space &operator=(const euclidean_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &x; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &x; }
	const_iterator end() const { return begin()+dim; }

	T x, y;
};/*}}}*/
template <class T> class euclidean_space<T,3>/*{{{*/
{
	typedef std::array<typename std::remove_const<T>::type,3> container_type;
public:
	static const int dim = 3;
	typedef T value_type;
	typedef euclidean_space coord_def_type;

	template <class U>
	euclidean_space &operator=(const euclidean_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	euclidean_space &operator=(const euclidean_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &x; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &x; }
	const_iterator end() const { return begin()+dim; }

	T x, y, z;
};/*}}}*/
template <class T> class euclidean_space<T,4>/*{{{*/
{
	typedef std::array<typename std::remove_const<T>::type,4> container_type;
public:
	static const int dim = 4;
	typedef T value_type;
	typedef euclidean_space coord_def_type;

	template <class U>
	euclidean_space &operator=(const euclidean_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	euclidean_space &operator=(const euclidean_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &x; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &x; }
	const_iterator end() const { return begin()+dim; }

	T x, y, z, w;
};/*}}}*/
template <class T> class euclidean_space<T,RUNTIME>/*{{{*/
{
	typedef std::vector<typename std::remove_const<T>::type> container_type;
public:
	static const int dim = RUNTIME;
	typedef T value_type;

	euclidean_space() {}

	euclidean_space(const euclidean_space &that)
		: m_coords(that.m_coords) {}

	euclidean_space(euclidean_space &&that)
		: m_coords(std::move(that.m_coords)) {}

	template <class U>
	euclidean_space(const euclidean_space<T,dim> &that) 
		: m_coords(that.m_coords) {}

	template <class U>
	euclidean_space(euclidean_space<T,dim> &&that) 
		: m_coords(std::move(that.m_coords)) {}

	template <class...ARGS>
	euclidean_space(const value_type &v1, ARGS &&...args)
		: m_coords{v1,value_type(args)...} {}

	euclidean_space(const dimension<1>::type &d) : m_coords(d[0]) {}

	template <class U>
	euclidean_space &operator=(const euclidean_space<T,dim> &that)
	{
		m_coords = that.m_coords;
		return *this;
	}

	template <class U>
	euclidean_space &operator=(euclidean_space<T,dim> &&that)
	{
		m_coords = std::move(that.m_coords);
		return *this;
	}

	euclidean_space &operator=(const euclidean_space &that)
	{
		m_coords = that.m_coords;
		return *this;
	}

	euclidean_space &operator=(euclidean_space &&that)
	{
		m_coords = std::move(that.m_coords);
		return *this;
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

private:
	container_type m_coords;
};/*}}}*/

template <class T, int D> 
struct is_view<euclidean_space_view<T,D>>
{
	static const bool value = true;
};


}} // namespace s3d::math

#endif
