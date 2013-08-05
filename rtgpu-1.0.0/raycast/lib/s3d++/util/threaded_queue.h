#ifndef S3D_UTIL_THREADED_QUEUE_H
#define S3D_UTIL_THREADED_QUEUE_H

#include <mutex>
#include <condition_variable>
#include <boost/optional.hpp>
#include <queue>
#include <limits>
#include <chrono>

namespace s3d
{

template <class T>
class threaded_queue
{
public:
	threaded_queue(size_t limit = std::numeric_limits<size_t>::max());

	void push(T value);

	template <class R, class P>
	bool push(T value, const std::chrono::duration<R,P> &timeout);

	template <class C, class D>
	bool push(T value, const std::chrono::time_point<C,D> &timeout);

	T pop();

	template <class R, class P>
	boost::optional<T> pop(const std::chrono::duration<R,P> &timeout);

	template <class C, class D>
	boost::optional<T> pop(const std::chrono::time_point<C,D> &timeout);

	size_t size() const;
	size_t limit() const { return m_limit; }

	bool empty() const;

	void wait_until_size(std::function<bool(size_t)> cond) const;

	template <class R, class P>
	bool wait_until_size(std::function<bool(size_t)> cond,
							const std::chrono::duration<R,P> &timeout) const;
	template <class C, class D>
	bool wait_until_size(std::function<bool(size_t)> cond,
							const std::chrono::time_point<C,D> &timeout) const;

	void wait_until_empty() const;

	template <class R, class P>
	bool wait_until_empty(const std::chrono::duration<R,P> &timeout) const;

	template <class C, class D>
	bool wait_until_empty(const std::chrono::time_point<C,D> &timeout) const;

private:
	typedef std::unique_lock<std::mutex> unique_lock;
	typedef std::queue<T> container_type;

	mutable std::mutex m_mtx;
	mutable std::condition_variable m_cond;

	container_type m_queue;

	size_t m_limit;
};

}

#include "threaded_queue.hpp"

#endif
