#ifndef S3D_MATH_PARAM_SPACE_H
#define S3D_MATH_PARAM_SPACE_H

namespace s3d { namespace math
{

template <class T, int D> class param_space_view/*{{{*/
{
	typedef memory_view<T,D> container_type;
public:
	typedef T value_type;
	static const int dim = D;

	param_space_view(T *data)
		: m_coords(data) {}

	param_space_view &operator=(const param_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	param_space_view &operator=(const param_space_view<U,D> &that)
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
template <class T> class param_space_view<T,1>/*{{{*/
{
public:
	static const int dim = 1;
	typedef T value_type;

	param_space_view(const param_space_view &that) 
		: u(that.u) {}

	param_space_view(T *data)
		: u(data[0])
	{
		assert(data);
	}

	param_space_view &operator=(const param_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	param_space_view &operator=(const param_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &u; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &u; }
	const_iterator end() const { return begin()+dim; }

	T &u;
};/*}}}*/
template <class T> class param_space_view<T,2>/*{{{*/
{
public:
	static const int dim = 2;
	typedef T value_type;

	param_space_view(const param_space_view &that) 
		: u(that.u), v(that.v) {}
	param_space_view(T *data)
		: u(data[0]), v(data[1])
	{
		assert(data);
	}

	param_space_view &operator=(const param_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	param_space_view &operator=(const param_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &u; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &u; }
	const_iterator end() const { return begin()+dim; }

	T &u, &v;
};/*}}}*/
template <class T> class param_space_view<T,3>/*{{{*/
{
public:
	static const int dim = 3;
	typedef T value_type;

	param_space_view(const param_space_view &that) 
		: u(that.u), v(that.v), w(that.w) {}

	param_space_view(T *data)
		: u(data[0]), v(data[1]), w(data[2])
	{
		assert(data);
	}

	param_space_view &operator=(const param_space_view &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	param_space_view &operator=(const param_space_view<U,dim> &that)
	{
		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &u; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &u; }
	const_iterator end() const { return begin()+dim; }

	T &u, &v, &w;
};/*}}}*/
template <class T> class param_space_view<T,RUNTIME>/*{{{*/
{
	typedef memory_view<T> container_type;
public:
	static const int dim = RUNTIME;
	typedef T value_type;

	param_space_view(const param_space_view &that) 
		: m_coords(that.m_coords) {}

	param_space_view(T *data, size_t size)
		: m_coords(data, size)
	{
		assert(data);
	}

	param_space_view &operator=(const param_space_view &that)
	{
		if(m_coords.size() != that.m_coords.size())
			throw std::runtime_error("Mismatched dimension");

		std::copy(that.begin(), that.end(), begin());
		return *this;
	}

	template <class U>
	param_space_view &operator=(const param_space_view<U,dim> &that)
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

	memory_view<T> m_coords;
};/*}}}*/

template <class T, int N=RUNTIME> class param_space/*{{{*/
{
	typedef std::array<typename std::remove_const<T>::type,N> container_type;
public:
	static_assert(N > 0, "Invalid dimension");

	static const int dim = N;
	typedef T value_type;

	param_space() {}

	template <class U>
	param_space(const param_space<U,N> &that) 
		: m_coords(that.m_coords) {}

	param_space(const dimension<1>::type &d)
	{
		if(d[0] != N)
			throw std::runtime_error("Mismatched dimension");
	}
	template <class...ARGS>
	param_space(const value_type &v1, ARGS &&...args)
		: m_coords((container_type){{v1, value_type(args)...}}) {}

	template <class U>
	param_space &operator=(const param_space<U,N> &that) 
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
template <class T> class param_space<T,1>/*{{{*/
{
public:
	static const int dim = 1;
	typedef T value_type;

	template <class U>
	param_space &operator=(const param_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	param_space &operator=(const param_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &u; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &u; }
	const_iterator end() const { return begin()+dim; }

	T u;
};/*}}}*/
template <class T> class param_space<T,2>/*{{{*/
{
public:
	static const int dim = 2;
	typedef T value_type;

	template <class U>
	param_space &operator=(const param_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	param_space &operator=(const param_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &u; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &u; }
	const_iterator end() const { return begin()+dim; }

	T u, v;
};/*}}}*/
template <class T> class param_space<T,3>/*{{{*/
{
public:
	static const int dim = 3;
	typedef T value_type;

	template <class U>
	param_space &operator=(const param_space<U,dim> &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	param_space &operator=(const param_space &that)
	{
		typedef typename std::remove_const<T>::type mutable_type;
		std::copy(that.begin(), that.end(), const_cast<mutable_type*>(begin()));
		return *this;
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	iterator begin() { return &u; }
	iterator end() { return begin()+dim; }

	const_iterator begin() const { return &u; }
	const_iterator end() const { return begin()+dim; }

	T u, v, w;
};/*}}}*/
template <class T> class param_space<T,RUNTIME>/*{{{*/
{
	typedef std::vector<typename std::remove_const<T>::type> container_type;
public:
	static const int dim = RUNTIME;
	typedef T value_type;

	param_space() {}

	param_space(const param_space &that)
		: m_coords(that.m_coords) {}

	param_space(param_space &&that)
		: m_coords(std::move(that.m_coords)) {}

	template <class U>
	param_space(const param_space<T,dim> &that) 
		: m_coords(that.m_coords) {}

	template <class U>
	param_space(param_space<T,dim> &&that) 
		: m_coords(std::move(that.m_coords)) {}

	template <class...ARGS>
	param_space(const value_type &v1, ARGS &&...args)
		: m_coords{v1,value_type(args)...} {}

	param_space(const dimension<1>::type &d) : m_coords(d[0]) {}

	template <class U>
	param_space &operator=(const param_space<T,dim> &that)
	{
		m_coords = that.m_coords;
		return *this;
	}

	template <class U>
	param_space &operator=(param_space<T,dim> &&that)
	{
		m_coords = std::move(that.m_coords);
		return *this;
	}

	param_space &operator=(const param_space &that)
	{
		m_coords = that.m_coords;
		return *this;
	}

	param_space &operator=(param_space &&that)
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
struct is_view<param_space_view<T,D>>
{
	static const bool value = true;
};

}}

#endif
