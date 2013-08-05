#ifndef S3D_UTIL_POINTER_H
#define S3D_UTIL_POINTER_H

#include <memory>

namespace s3d
{
	inline void null_deleter(const void *) {}

	template <class T, class...ARGS> 
	auto make_unique(ARGS &&...args)
		-> typename std::enable_if<!std::is_array<T>::value,
				std::unique_ptr<T> >::type
	{
		return std::unique_ptr<T>(new T(std::forward<ARGS>(args)...));
	}

	template <class T> 
	std::shared_ptr<T> make_shared(std::unique_ptr<T> ptr)
	{
		return std::shared_ptr<T>(std::move(ptr));
	}

	template <class T> 
	std::shared_ptr<T> to_pointer(std::reference_wrapper<T> ref)
	{
		return std::shared_ptr<T>(&ref.get(), null_deleter);
	}

	template <class T>
	std::unique_ptr<T> to_pointer(std::unique_ptr<T> ptr)
	{
		return std::move(ptr);
	}

	template <class T>
	std::shared_ptr<T> to_pointer(std::shared_ptr<T> ptr)
	{
		return ptr;
	}
}

namespace std
{
	template <class TO, class FROM>
	std::unique_ptr<TO> static_pointer_cast(std::unique_ptr<FROM> ptr)
	{
		return std::unique_ptr<TO>(static_cast<TO *>(ptr.release()));
	}

	template <class TO, class FROM>
	std::unique_ptr<TO> const_pointer_cast(std::unique_ptr<FROM> ptr)
	{
		return std::unique_ptr<TO>(const_cast<TO *>(ptr.release()));
	}

	template <class TO, class FROM>
	std::unique_ptr<TO> reinterpret_pointer_cast(std::unique_ptr<FROM> ptr)
	{
		return std::unique_ptr<TO>(reinterpret_cast<TO *>(ptr.release()));
	}

	template <class TO, class FROM>
	std::unique_ptr<TO> dynamic_pointer_cast(std::unique_ptr<FROM> ptr)
	{
		if(auto *p = dynamic_cast<TO *>(ptr.get()))
		{
			std::unique_ptr<TO> ret(p);
			ptr.release();
			return std::move(ret);
		}
		else
			return std::unique_ptr<TO>(NULL);
	}
}


#endif
