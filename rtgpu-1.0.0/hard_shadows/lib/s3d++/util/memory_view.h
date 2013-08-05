#ifndef S3D_UTIL_MEMORY_VIEW_H
#define S3D_UTIL_MEMORY_VIEW_H

#include <iterator>

namespace s3d
{

template <class T, int M=-1> class memory_view 
{
public:
	typedef T value_type;
	static const int dim = M;

	memory_view(T *data, size_t m=M)
		: m_data(data)
	{
		assert(m == M);
	}

	typedef T *iterator;
	typedef const T *const_iterator;

	typedef std::reverse_iterator<iterator> reverse_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

	T &operator[](int i)
	{
		assert(i >= 0 && (size_t)i < size());
		return m_data[i];
	}
	const T &operator[](int i) const
	{
		assert(i >= 0 && (size_t)i < size());
		return m_data[i];
	}

	iterator begin() { return data(); }
	iterator end() { return data()+size(); }

	const_iterator begin() const { return cbegin(); }
	const_iterator end() const { return cend(); }

	const_iterator cbegin() { return data(); }
	const_iterator cend() { return data()+size(); }

	const_iterator cbegin() const { return data(); }
	const_iterator cend() const { return data()+size(); }

	reverse_iterator rbegin() { return reverse_iterator(end()); }
	reverse_iterator rend() { return reverse_iterator(begin()); }

	const_reverse_iterator rbegin() const { return reverse_iterator(end()); }
	const_reverse_iterator rend() const { return reverse_iterator(begin()); }

	const_reverse_iterator crbegin() { return reverse_iterator(end()); }
	const_reverse_iterator crend() { return reverse_iterator(begin()); }

	const_reverse_iterator crbegin() const { return reverse_iterator(end()); }
	const_reverse_iterator crend() const { return reverse_iterator(begin()); }

	size_t size() const { return M; }
	const T *data() const { return m_data; }
	T *data() { return m_data; }

private:
	T *m_data;
};

template <class T> class memory_view<T>
{
public:
	typedef T value_type;
	static const int dim = -1;

	memory_view(T *data, size_t size)
		: m_data(data), m_size(size) {}

	memory_view(const memory_view &that) = default;
	memory_view &operator=(const memory_view &that) = default;

	typedef T *iterator;
	typedef const T *const_iterator;

	typedef std::reverse_iterator<iterator> reverse_iterator;
	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

	T &operator[](int i)
	{
		assert(i >= 0 && (size_t)i < m_size);
		return m_data[i];
	}
	const T &operator[](int i) const
	{
		assert(i >= 0 && (size_t)i < m_size);
		return m_data[i];
	}

	iterator begin() { return data(); }
	iterator end() { return data()+size(); }

	const_iterator begin() const { return data(); }
	const_iterator end() const { return data()+size(); }

	const_iterator cbegin() { return data(); }
	const_iterator cend() { return data()+size(); }

	const_iterator cbegin() const { return data(); }
	const_iterator cend() const { return data()+size(); }

	reverse_iterator rbegin() { return reverse_iterator(end()); }
	reverse_iterator rend() { return reverse_iterator(begin()); }

	const_reverse_iterator rbegin() const { return reverse_iterator(end()); }
	const_reverse_iterator rend() const { return reverse_iterator(begin()); }

	const_reverse_iterator crbegin() { return reverse_iterator(end()); }
	const_reverse_iterator crend() { return reverse_iterator(begin()); }

	const_reverse_iterator crbegin() const { return reverse_iterator(end()); }
	const_reverse_iterator crend() const { return reverse_iterator(begin()); }

	size_t size() const { return m_size; }
	const T *data() const { return m_data; }
	T *data() { return m_data; }

private:
	T *m_data;
	size_t m_size;
};

}

#endif
