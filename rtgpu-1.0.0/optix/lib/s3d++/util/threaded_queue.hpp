#include "gcc.h"

#if GCC_VERSION < 40500
namespace std
{
	enum cv_status { no_timeout, timeout };
}
#endif

namespace s3d 
{

template <class T>
threaded_queue<T>::threaded_queue(size_t limit)
	: m_limit(limit)
{
}

template <class T>
void threaded_queue<T>::push(T value)
{
	unique_lock lk(m_mtx);

	while(m_queue.size() >= m_limit)
		m_cond.wait(lk);

	m_queue.push(std::move(value));

	m_cond.notify_all();
}

template <class T>
template <class C, class D>
bool threaded_queue<T>::push(T value, const std::chrono::time_point<C,D> &timeout)
{
	unique_lock lk(m_mtx);

	while(m_queue.size() >= m_limit)
	{
		if(!m_cond.wait_until(lk, timeout))
			break;
	}

	if(m_queue.size() >= m_limit)
		return false;

	m_queue.push(std::move(value));

	m_cond.notify_all();

	return true;
}

template <class T>
template <class R, class P>
bool threaded_queue<T>::push(T value, const std::chrono::duration<R,P> &timeout)
{
	return push(std::move(value), std::chrono::system_clock::now() + timeout);
}

template <class T>
T threaded_queue<T>::pop()
{
	unique_lock lk(m_mtx);

	while(m_queue.empty())
		m_cond.wait(lk);

	T value = std::move(m_queue.front());
	m_queue.pop();

	m_cond.notify_all();

	return value; // RVO kicks in
}

template <class T>
template <class C, class D>
boost::optional<T> threaded_queue<T>::pop(
						const std::chrono::time_point<C,D> &timeout)
{
	unique_lock lk(m_mtx);

	while(m_queue.empty())
	{
		if(m_cond.wait_until(lk, timeout) == std::cv_status::timeout)
			break;
	}

	if(m_queue.empty())
		return boost::none;

	T value = std::move(m_queue.front());
	m_queue.pop();

	m_cond.notify_all();

	return value; // RVO kicks in
}

template <class T>
template <class R, class P>
boost::optional<T> threaded_queue<T>::pop(const std::chrono::duration<R,P> &timeout)
{
	return pop(std::chrono::system_clock::now() + timeout);
}

template <class T>
size_t threaded_queue<T>::size() const
{
	unique_lock lk(m_mtx);
	return m_queue.size();
}

template <class T>
bool threaded_queue<T>::empty() const
{
	unique_lock lk(m_mtx);
	return m_queue.empty();
}

template <class T>
void threaded_queue<T>::wait_until_size(std::function<bool(size_t)> cond) const
{
	unique_lock lk(m_mtx);

	while(!cond(m_queue.size()))
		m_cond.wait(lk);
}

template <class T>
template <class C, class D>
bool threaded_queue<T>::wait_until_size(std::function<bool(size_t)> cond,
							const std::chrono::time_point<C,D> &timeout) const
{
	unique_lock lk(m_mtx);

	while(!cond(m_queue.size()))
		if(m_cond.wait_until(lk, timeout) == std::cv_status::timeout)
			break;

	return cond(m_queue.size());
}

template <class T>
template <class R, class P>
bool threaded_queue<T>::wait_until_size(std::function<bool(size_t)> cond,
							const std::chrono::duration<R,P> &timeout) const
{
	return wait_until_size(cond, std::chrono::system_clock::now() + timeout);
}

template <class T>
void threaded_queue<T>::wait_until_empty() const
{
	unique_lock lk(m_mtx);

	while(!m_queue.empty())
		m_cond.wait(lk);
}

template <class T>
template <class C, class D>
bool threaded_queue<T>::wait_until_empty(
					const std::chrono::time_point<C,D> &timeout) const
{
	unique_lock lk(m_mtx);

	while(!m_queue.empty())
		if(!m_cond.wait_until(lk, timeout))
			break;

	return m_queue.empty();
}

template <class T>
template <class R, class P>
bool threaded_queue<T>::wait_until_empty(
							const std::chrono::duration<R,P> &timeout) const
{
	return wait_until_empty(std::chrono::system_clock::now() + timeout);
}

}
