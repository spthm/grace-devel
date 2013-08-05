#ifndef S3D_MATH_SIZE_SPACE_H
#define S3D_MATH_SIZE_SPACE_H

namespace s3d { namespace math
{

template <class T, int D> class size_space_view/*{{{*/
{
	typedef memory_view<T,D> container_type;
public:
	typedef T value_type;
	static const int dim = D;

	size_space_view(T *data)
		: m_coords(data) {}

	size_space_view &operator=(const size_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	size_space_view &operator=(const size_space_view<U,D> &that)
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
template <class T> class size_space_view<T,1>/*{{{*/
{
public:
	static const int dim = 1;
	typedef T value_type;

	size_space_view(const size_space_view &that) 
		: w(that.w) {}

	size_space_view(T *data)
		: w(data[0])
	{
		assert(data);
	}

	size_space_view &operator=(const size_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	size_space_view &operator=(const size_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &w; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &w; }
	const_iterator end() const { return begin()+dim; }

	T &w;
};/*}}}*/
template <class T> class size_space_view<T,2>/*{{{*/
{
public:
	static const int dim = 2;
	typedef T value_type;

	size_space_view(const size_space_view &that) 
		: w(that.w), h(that.h) {}
	size_space_view(T *data)
		: w(data[0]), h(data[1])
	{
		assert(data);
	}

	size_space_view &operator=(const size_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	size_space_view &operator=(const size_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &w; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &w; }
	const_iterator end() const { return begin()+dim; }

	T &w, &h;
};/*}}}*/
template <class T> class size_space_view<T,3>/*{{{*/
{
public:
	static const int dim = 3;
	typedef T value_type;

	size_space_view(const size_space_view &that) 
		: w(that.w), h(that.h), d(that.d) {}

	size_space_view(T *data)
		: w(data[0]), h(data[1]), d(data[2])
	{
		assert(data);
	}

	size_space_view &operator=(const size_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	size_space_view &operator=(const size_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &w; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &w; }
	const_iterator end() const { return begin()+dim; }

	T &w, &h, &d;
};/*}}}*/
template <class T> class size_space_view<T,4>/*{{{*/
{
public:
	static const int dim = 4;
	typedef T value_type;

	size_space_view(const size_space_view &that) 
		: w(that.w), h(that.h), d(that.d), s(that.s) {}

	size_space_view(T *data)
		: w(data[0]), h(data[1]), d(data[2]), s(data[3])
	{
		assert(data);
	}

	size_space_view &operator=(const size_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	size_space_view &operator=(const size_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &w; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &w; }
	const_iterator end() const { return begin()+dim; }

	T &w, &h, &d, &s;
};/*}}}*/
template <class T> class size_space_view<T,RUNTIME>/*{{{*/
{
	typedef memory_view<T> container_type;
public:
	static const int dim = RUNTIME;
	typedef T value_type;

	size_space_view(const size_space_view &that) 
		: m_coords(that.m_coords) {}

	size_space_view(T *data, size_t size)
		: m_coords(data, size)
	{
		assert(data);
	}

	size_space_view &operator=(const size_space_view &that)
	{
		if(m_coords.size() != that.m_coords.size())
			throw std::runtime_error("Miwmatched dimension");

		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	size_space_view &operator=(const size_space_view<U,dim> &that)
	{
		if(m_coords.size() != that.m_coords.size())
			throw std::runtime_error("Miwmatched dimension");

		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

	memory_view<T> m_coords;
};/*}}}*/

template <class T, int N=RUNTIME> class size_space/*{{{*/
{
	typedef std::array<typename std::remove_const<T>::type,N> container_type;
public:
	static_assert(N > 0, "Invalid dimension");

	static const int dim = N;
	typedef T value_type;

	size_space() {}

	template <class U>
	size_space(const size_space<U,N> &that) 
		: m_coords(that.m_coords) {}

	size_space(const dimension<1>::type &d)
	{
		if(d[0] != N)
			throw std::runtime_error("Miwmatched dimension");
	}
	template <class...ARGS>
	size_space(const value_type &v1, ARGS &&...argw)
		: m_coords((container_type){{v1, value_type(argw)...}}) {}

	template <class U>
	size_space &operator=(const size_space<U,N> &that) 
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
template <class T> class size_space<T,1>/*{{{*/
{
public:
	static const int dim = 1;
	typedef T value_type;

	template <class U>
	size_space &operator=(const size_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	size_space &operator=(const size_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &w; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &w; }
	const_iterator end() const { return begin()+dim; }

	T w;
};/*}}}*/
template <class T> class size_space<T,2>/*{{{*/
{
public:
	static const int dim = 2;
	typedef T value_type;

	template <class U>
	size_space &operator=(const size_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	size_space &operator=(const size_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &w; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &w; }
	const_iterator end() const { return begin()+dim; }

	T w, h;
};/*}}}*/
template <class T> class size_space<T,3>/*{{{*/
{
public:
	static const int dim = 3;
	typedef T value_type;

	template <class U>
	size_space &operator=(const size_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	size_space &operator=(const size_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &w; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &w; }
	const_iterator end() const { return begin()+dim; }

	T w, h, d;
};/*}}}*/
template <class T> class size_space<T,4>/*{{{*/
{
public:
	static const int dim = 4;
	typedef T value_type;

	template <class U>
	size_space &operator=(const size_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	size_space &operator=(const size_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &w; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &w; }
	const_iterator end() const { return begin()+dim; }

	T w, h, d, s;
};/*}}}*/
template <class T> class size_space<T,RUNTIME>/*{{{*/
{
	typedef std::vector<typename std::remove_const<T>::type> container_type;
public:
	static const int dim = RUNTIME;
	typedef T value_type;

	size_space() {}

	size_space(const size_space &that)
		: m_coords(that.m_coords) {}

	size_space(size_space &&that)
		: m_coords(std::move(that.m_coords)) {}

	template <class U>
	size_space(const size_space<T,dim> &that) 
		: m_coords(that.m_coords) {}

	template <class U>
	size_space(size_space<T,dim> &&that) 
		: m_coords(std::move(that.m_coords)) {}

	template <class...ARGS>
	size_space(const value_type &v1, ARGS &&...argw)
		: m_coords{v1,value_type(argw)...} {}

	size_space(const dimension<1>::type &d) : m_coords(d[0]) {}

	template <class U>
	size_space &operator=(const size_space<T,dim> &that)
	{
		m_coords = that.m_coords;
		return *this;
	}

	template <class U>
	size_space &operator=(size_space<T,dim> &&that)
	{
		m_coords = std::move(that.m_coords);
		return *this;
	}

	size_space &operator=(const size_space &that)
	{
		m_coords = that.m_coords;
		return *this;
	}

	size_space &operator=(size_space &&that)
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
struct is_view<size_space_view<T,D>>
{
	static const bool value = true;
};

}} // namespace s3d::math

#endif
