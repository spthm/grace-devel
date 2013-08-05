#ifndef S3D_MATH_MATRIX_SPACE_H
#define S3D_MATH_MATRIX_SPACE_H

#include <boost/utility/base_from_member.hpp>
#include "../mpl/create_range.h"
#include "vector_view.h"

namespace s3d { namespace math
{

// matrix_space_view

template <class T, int M, int N>
class matrix_space_view/*{{{*/
{
	typedef std::array<VectorView<T>, M> container_type;
public:
	static_assert(M > 0, "Invalid row count");
	static_assert(N > 0, "Invalid column count");

	typedef VectorView<T> value_type;

	matrix_space_view(T *data, size_t stride=N)
		: m_coords(init_coords(data,stride,mpl::create_range<int,0,M>::type()))
	{
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

protected:
	container_type m_coords;
private:

	template <int...D>
	static container_type init_coords(T *data, size_t stride, 
									  mpl::vector_c<int,D...>)
	{
		return {{(data+stride*D)...}};
	}
};/*}}}*/

template <class T, int M>
class matrix_space_view<T,M,RUNTIME>/*{{{*/
{
	typedef std::array<VectorView<T>, M> container_type;
public:
	static_assert(M > 0, "Invalid row count");

	typedef VectorView<T> value_type;

	matrix_space_view(T *data, size_t N, size_t stride=(size_t)-1)
	   : m_coords(init_coords(data,(stride==(size_t)-1 ? N : stride),
	   						  N,typename mpl::create_range<int,0,M>::type()))
	{
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

protected:
	container_type m_coords;

	void reassign(T *data, size_t stride)
	{
		for(size_t i=0; i<m_coords.size(); ++i)
			m_coords[i].reassign(data+stride*i, stride);
	}
private:

	template <int...D>
	static container_type init_coords(T *data, size_t stride, size_t N,
									  mpl::vector_c<int,D...>)
	{
		return container_type{ { {(data+stride*D),N}... } };
	}
};/*}}}*/

template <class T, int N>
class matrix_space_view<T,RUNTIME, N>/*{{{*/
{
	typedef std::vector<VectorView<T,N>> container_type;
public:
	static_assert(N > 0, "Invalid row count");

	typedef VectorView<T> value_type;

	matrix_space_view(const matrix_space_view &that) = default;
	matrix_space_view(matrix_space_view &&that)
		: m_coords(std::move(that.m_coords)) {}

	matrix_space_view(T *data, size_t M, size_t stride=N)
	{
		m_coords.reserve(M);
		for(size_t i=0; i<M; ++i)
			m_coords.emplace_back(data+i*stride);
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

protected:
	container_type m_coords;
};/*}}}*/

template <class T>
class matrix_space_view<T,RUNTIME,RUNTIME>/*{{{*/
{
	typedef std::vector<VectorView<T>> container_type;
public:
	typedef VectorView<T> value_type;

	matrix_space_view(const matrix_space_view &that) = default;
	matrix_space_view(matrix_space_view &&that)
		: m_coords(std::move(that.m_coords)) {}

	matrix_space_view(T *data, size_t M, size_t N, size_t stride=(size_t)-1)
	{
		if(stride == (size_t)-1)
			stride = N;

		m_coords.reserve(M);
		for(size_t i=0; i<M; ++i)
			m_coords.emplace_back(data+i*stride, N);
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

protected:
	container_type m_coords;
};/*}}}*/

// matrix_space

template <class T, int M, int N>
class matrix_space/*{{{*/
{
	typedef std::array<Vector<T,N>,M> container_type;
public:
	typedef Vector<T> value_type;

	matrix_space() {}

	template <class...ARGS>
	matrix_space(const value_type &v1, ARGS &&...args)
		: m_coords((container_type){{v1, args...}}) {}

	matrix_space(const dimension<1>::type &d)
	{
		static_assert(M==N, "Must specify 2 dimensions");

		if(d[0] != M || d[1] != N)
			throw std::runtime_error("Mismatched dimension");
	}

	matrix_space(const dimension<2>::type &d)
	{
		if(d[0] != M || d[1] != N)
			throw std::runtime_error("Mismatched dimension");
	}

	typedef typename container_type::iterator iterator;
	typedef typename container_type::const_iterator const_iterator;

	iterator begin() { return m_coords.begin(); }
	iterator end() { return m_coords.end(); }

	const_iterator begin() const { return m_coords.begin(); }
	const_iterator end() const { return m_coords.end(); }

protected:
	container_type m_coords;
};/*}}}*/

template <class T, int N> 
class matrix_space<T,RUNTIME,N>/*{{{*/
{
	typedef std::vector<Vector<T,N>> container_type;
public:
	typedef Vector<T> value_type;

	matrix_space() {}

	template <class...ARGS>
	matrix_space(const value_type &v1, ARGS &&...args)
		: m_coords{v1,args...} {}

	matrix_space(const dimension<1>::type &d)
		: m_coords(d[0])
	{
		assert(m_coords.size() == d[0]);
#ifndef NDEBUG
		for(std::size_t i=0; i<m_coords.size(); ++i)
			assert(m_coords[i].size() == N);
#endif
	}

	matrix_space(const dimension<2>::type &d)
		: m_coords(d[0])
	{
		if(d[1] != N)
			throw std::runtime_error("Mismatched dimension");

		assert(m_coords.size() == d[0]);
	}

	matrix_space(const matrix_space &that) = default;
	matrix_space(matrix_space &&that)
		: m_coords(std::move(that.m_coords))
	{
	}

	matrix_space &operator=(const matrix_space &that) = default;
	matrix_space &operator=(matrix_space &&that)
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

protected:
	container_type m_coords;
};/*}}}*/

template <class T, int M> 
class matrix_space<T,M,RUNTIME>/*{{{*/
	: public boost::base_from_member<std::vector<T>>
	, public matrix_space_view<T,M,RUNTIME>
{
	typedef matrix_space_view<T,M,RUNTIME> base_view;
	typedef boost::base_from_member<std::vector<T>> base_values;
	using base_values::member;

public:
	using base_view::begin;
	using base_view::end;

	matrix_space() : base_view(NULL, 0, 0) {}

	matrix_space(const matrix_space &that)
		: base_values(that.member)
		, base_view(&member[0], that.begin()->size())
	{
	}

	matrix_space(matrix_space &&that)
		: base_values(std::move(that.member))
		, base_view(&member[0], that.begin()->size())
	{
		that.reassign(NULL, 0);
	}

	template <int N, class...ARGS>
	matrix_space(const Vector<T,N> &v1, const ARGS &...args)
		: base_values(M*v1.size())
		, base_view(&member[0], v1.size())

	{
		copy_values(&member[0], v1, args...);
	}

	matrix_space(const dimension<1>::type &d)
		: base_values(M*d[0])
		, base_view(&member[0], d[0])
	{
	}

	matrix_space(const dimension<2>::type &d)
		: base_values(M*d[1])
		, base_view(&member[0], d[1])
	{
		assert(d[0] == M);
		if(d[0] != M)
			throw std::runtime_error("Mismatched dimension");
	}

	matrix_space &operator=(matrix_space &&that)
	{
		size_t N = that.begin()->size();

		if(member.empty())
		{
			member = std::move(that.member);
			this->reassign(&member[0], N);
		}
		else
		{
			assert(!begin()->empty());

			if(begin()->size() != N)
				throw std::runtime_error("Mismatched dimensions");

			assert(&begin()->operator[](0) == &member[0]);

			assert(member.size() == that.member.size());
			member = std::move(that.member);
			this->reassign(&member[0], N);
		}

		that.reassign(NULL, 0);

		return *this;
	}

	matrix_space &operator=(const matrix_space &that)
	{
		if(member.empty() && that.member.empty())
			return *this;

		assert(!that.begin()->empty());
		size_t N = that.begin()->size();

		// were we an "empty" matrix?
		if(member.empty())
		{
			// must init m_coords
			member = that.member;
			this->reassign(&member[0], N);
		}
		else
		{
			assert(!begin()->empty());

			if(begin()->size() != N)
				throw std::runtime_error("Mismatched dimensions");

			assert(&begin()->operator[](0) == &member[0]);

			// member internal data must not be reallocated somewhere else!
			assert(member.size() == that.member.size());
			member = that.member;
		}

		return *this;
	}

private:
	template <int N>
	static T *copy_values(T *out, const Vector<T,N> &v1)/*{{{*/
	{
		std::copy(v1.begin(), v1.end(), out);
		return out += v1.size();
	}/*}}}*/

	template <int N, int P, class...ARGS>
	static T *copy_values(T *out, const Vector<T,N> &v1, const Vector<T,P> &v2, /*{{{*/
						  const ARGS &...args)
	{
		static_assert(N==RUNTIME || P==RUNTIME || N==P,
					  "Vectors must have same size");

		if(v1.size() != v2.size())
			throw std::runtime_error("Vectors must have same size");

		copy_values(copy_values(out, v1), v2, args...);
	}/*}}}*/
};/*}}}*/

template <class T> 
class matrix_space<T,RUNTIME,RUNTIME>/*{{{*/
	: public boost::base_from_member<std::vector<T>>
	, public matrix_space_view<T,RUNTIME,RUNTIME>
{
	typedef boost::base_from_member<std::vector<T>> base_values;
	typedef matrix_space_view<T,RUNTIME,RUNTIME> base_view;

	using base_values::member;
public:
	using base_view::begin;
	using base_view::end;

	matrix_space() : base_view(NULL, 0, 0, 0) {}

	matrix_space(matrix_space &&that)
		: base_view(std::move(that))
	{
		member = std::move(that.member);
	}

	matrix_space(const matrix_space &that)
		: base_values(that.member)
		, base_view(&member[0], that.m_coords.size(),
					that.m_coords.empty()?0:that.m_coords[0].size())
	{
	}

	template <int N, class...ARGS>
	matrix_space(const Vector<T,N> &v1, const ARGS &...args)
		: base_values((1+sizeof...(args))*v1.size())
		, base_view(&member[0], 1+sizeof...(args), v1.size())
	{
		copy_values(&member[0], v1, args...);
	}

	matrix_space(const dimension<1>::type &d)
		: base_values(d[0]*d[0])
		, base_view(&member[0], d[0], d[0])
	{
	}

	matrix_space(const dimension<2>::type &d)
		: base_values(d[0]*d[1])
		, base_view(&member[0], d[0], d[1])
	{
	}

	matrix_space &operator=(matrix_space &&that)
	{
		size_t this_size = std::distance(begin(), end()),
			   that_size = std::distance(that.begin(), that.end());

		if(this_size!=0 && (that_size==0 ||
			(that_size!=this_size || begin()->size() != that.begin()->size())))
	    {
	    	throw std::runtime_error("Mismatched dimensions");
 	    }

 	    member = std::move(that.member);
 	    this->m_coords = std::move(that.m_coords);

		assert(member.empty() || &*begin()->begin() == &member[0]);
		return *this;
	}

	matrix_space &operator=(const matrix_space &that)
	{
		size_t this_size = std::distance(begin(), end()),
			   that_size = std::distance(that.begin(), that.end());

		if(this_size==0 && that_size==0)
			return *this;

		if(this_size!=0 && (that_size==0 ||
			(that_size!=this_size || begin()->size() != that.begin()->size())))
	    {
	    	throw std::runtime_error("Mismatched dimensions");
 	    }

		assert(that_size!=0);

 	    size_t N = that.begin()->size();
		assert(N != 0);

		// were we an "empty" matrix?
		if(member.empty())
		{
			// must init m_coords
			member = that.member;
			auto *m = &member[0];
			assert(this->m_coords.empty());
			this->m_coords.clear(); // só pra garantir
			this->m_coords.reserve(that.m_coords.size());
			for(size_t i=0; i<that.m_coords.size(); ++i, m += N)
				this->m_coords.emplace_back(m, N);
		}
		else
		{
			assert(!this->m_coords[0].empty());
			assert(&this->m_coords[0][0] == &member[0]);

			// member internal data must not be reallocated somewhere else!
			assert(member.size() == that.member.size());
			member = that.member;
		}

		return *this;
	}
private:
	template <int N>
	static T *copy_values(T *out, const Vector<T,N> &v1)/*{{{*/
	{
		std::copy(v1.begin(), v1.end(), out);
		return out += v1.size();
	}/*}}}*/

	template <int N, int P, class...ARGS>
	static T *copy_values(T *out, const Vector<T,N> &v1, const Vector<T,P> &v2, /*{{{*/
						  const ARGS &...args)
	{
		static_assert(N==RUNTIME || P==RUNTIME || N==P,
					  "Vectors must have same size");

		if(v1.size() != v2.size())
			throw std::runtime_error("Vectors must have same size");

		copy_values(copy_values(out, v1), v2, args...);
	}/*}}}*/
};/*}}}*/


}} // namespace s3d::math

#endif
