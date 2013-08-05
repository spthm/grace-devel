#ifndef S3D_FLOAT4_H
#define S3D_FLOAT4_H

#include <boost/operators.hpp>
#ifdef __SSE__
#	include <xmmintrin.h>
#endif

#ifdef __SSE3__
#	include <pmmintrin.h>
#endif
#include <memory>

namespace s3d { namespace math {

class v4f 
	: public boost::arithmetic<v4f,
			 boost::andable<v4f,
			 boost::xorable<v4f,
			 boost::orable<v4f>>>>
{
public:
	v4f() {}
	v4f(float a) : m_data(_mm_load1_ps(&a)) {}
	v4f(float a[4]) : m_data(_mm_load_ps(a)) {}
	v4f(float a0, float a1, float a2, float a3)
		: m_data(_mm_set_ps(a3, a2, a1, a0)) {}
	v4f(const __m128 &a) : m_data(a) {}

	operator __m128() const { return m_data; }

	void *operator new(size_t size)/*{{{*/
	{
		return _mm_malloc(size, 16);
	}/*}}}*/
	void *operator new(size_t size, void *ptr)/*{{{*/
	{
		if(((ptrdiff_t)ptr & 0xF) != 0)
			throw std::bad_alloc();
		return ptr;
	}/*}}}*/

	void operator delete(void *p)/*{{{*/
	{
		_mm_free(p);
	}/*}}}*/
	void *operator new[](size_t size)/*{{{*/
	{
		return _mm_malloc(size, 16);
	}/*}}}*/
	void operator delete[](void *p)/*{{{*/
	{
		_mm_free(p);
	}/*}}}*/

	void convert_to(float a[4]) const/*{{{*/
	{
		_mm_store_ps(a, m_data);
	}/*}}}*/
	void convert_to(float &a0, float &a1, float &a2, float &a3) const/*{{{*/
	{
		float aux[4] __attribute__((aligned(16)));
		convert_to(aux);
		a0 = aux[0];
		a1 = aux[1];
		a2 = aux[2];
		a3 = aux[3];
	}/*}}}*/

	v4f &operator+=(const v4f &a)/*{{{*/
	{
		m_data = _mm_add_ps(m_data,a.m_data);
		return *this;
	}/*}}}*/
	v4f &operator-=(const v4f &a)/*{{{*/
	{
		m_data = _mm_sub_ps(m_data,a.m_data);
		return *this;
	}/*}}}*/
	v4f &operator*=(const v4f &a)/*{{{*/
	{
		m_data = _mm_mul_ps(m_data,a.m_data);
		return *this;
	}/*}}}*/
	v4f &operator/=(const v4f &a)/*{{{*/
	{
		m_data = _mm_div_ps(m_data,a.m_data);
		return *this;
	}/*}}}*/
	v4f &operator |=(const v4f &a)/*{{{*/
	{
		m_data = _mm_or_ps(m_data,a.m_data);
		return *this;
	}/*}}}*/
	v4f &operator &=(const v4f &a)/*{{{*/
	{
		m_data = _mm_and_ps(m_data,a.m_data);
		return *this;
	}/*}}}*/
	v4f &operator ^=(const v4f &a)/*{{{*/
	{
		m_data = _mm_xor_ps(m_data,a.m_data);
		return *this;
	}/*}}}*/

	v4f operator <(const v4f &that) const/*{{{*/
	{
		return _mm_cmplt_ps(m_data, that.m_data);
	}/*}}}*/
	v4f operator >(const v4f &that) const/*{{{*/
	{
		return _mm_cmpgt_ps(m_data, that.m_data);
	}/*}}}*/
	v4f operator <=(const v4f &that) const/*{{{*/
	{
		return _mm_cmple_ps(m_data, that.m_data);
	}/*}}}*/
	v4f operator >=(const v4f &that) const/*{{{*/
	{
		return _mm_cmpge_ps(m_data, that.m_data);
	}/*}}}*/
	v4f operator !=(const v4f &that) const/*{{{*/
	{
		return _mm_cmpneq_ps(m_data, that.m_data);
	}/*}}}*/

	friend v4f sqrt(const v4f &a)/*{{{*/
	{
		return _mm_sqrt_ps(a);
	}/*}}}*/
#ifdef SIMDMATH_H
	friend v4f atan2(const v4f &y, const v4f &x)/*{{{*/
	{
		return _atan2f4(y,x);
	}/*}}}*/
	friend v4f atan(const v4f &a)/*{{{*/
	{
		return _atanf4(a);
	}/*}}}*/
	friend v4f sin(const v4f &a)/*{{{*/
	{
		return _sinf4(a);
	}/*}}}*/
	friend v4f cos(const v4f &a)/*{{{*/
	{
		return _cosf4(a);
	}/*}}}*/
	friend void sincos(v4f &s, v4f &c, const v4f &a)/*{{{*/
	{
		_sincosf4(a,&s.m_data, &c.m_data);
	}/*}}}*/
	friend v4f exp(const v4f &a)/*{{{*/
	{
		return _expf4(a);
	}/*}}}*/
	friend v4f floor(const v4f &a)/*{{{*/
	{
		return _floorf4(a);
	}/*}}}*/
	friend v4f abs(const v4f &a)/*{{{*/
	{
		return _fabsf4(a);
	}/*}}}*/
	friend v4f pow(const v4f &a, const v4f &b)/*{{{*/
	{
		return _powf4(a,b);
	}/*}}}*/
#endif
	friend v4f inv(const v4f &a)/*{{{*/
	{
		return _mm_rcp_ps(a);
	}/*}}}*/
	friend v4f inv_sqrt(const v4f &a)/*{{{*/
	{
		return _mm_rsqrt_ps(a);
	}/*}}}*/
	friend v4f zero()/*{{{*/
	{
		return _mm_setzero_ps();
	}/*}}}*/
	friend v4f andnot(const v4f &a, const v4f &mask)/*{{{*/
	{
		return _mm_andnot_ps(mask, a);
	}/*}}}*/

private:
	__m128 m_data;
};

} // namespace math

using math::v4f;

} // namespace s3d

namespace std
{
	template<>
	class allocator<s3d::v4f>
	{
	public:
		typedef size_t size_type;
		typedef ptrdiff_t difference_type;
		typedef s3d::v4f *pointer;
		typedef const s3d::v4f *const_pointer;
		typedef s3d::v4f &reference;
		typedef const s3d::v4f &const_reference;
		typedef s3d::v4f value_type;

		template <class T>
		struct rebind
			{ typedef allocator<T> other; };

		pointer address(reference __x) const { return &__x; }
		const_pointer address(const_reference __x) const { return &__x; }

		pointer allocate(size_type __n, const void * = 0)
		{
			if(__builtin_expect(__n > this->max_size(), false))
				throw std::bad_alloc();

			return (pointer)_mm_malloc(__n*sizeof(value_type), 16);
		}

		void deallocate(pointer __p, size_type)
			{ _mm_free(__p); }

		size_type max_size() const throw()
			{ return size_t(-1) / sizeof(value_type); }

		void construct(pointer p, const value_type &val)
			{ ::new((void *)p) value_type(val); }

		void destroy(pointer p) { p->~value_type(); }
	};
} // namespace std


#endif
