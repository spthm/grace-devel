#ifndef S3D_COLOR_CAST_DEF_H
#define S3D_COLOR_CAST_DEF_H

namespace s3d { namespace color
{

template <class FROM, class TO>
struct cast_def
{
	// FROM can be converted to TO explicitly
	template <class DUMMY=int>
	static auto map(const FROM &c)
		-> typename std::enable_if<sizeof(DUMMY) &&
			!std::is_arithmetic<TO>::value &&
			is_explicitly_convertible<FROM,TO>::value, TO>::type
	{
		return TO(c);
	}

	// default: convert through radiance
	template <class DUMMY=int>
	static auto map(const FROM &c)
		-> typename std::enable_if<sizeof(DUMMY) &&
			!std::is_arithmetic<TO>::value &&
			!std::is_arithmetic<FROM>::value &&
			!is_explicitly_convertible<FROM,TO>::value, TO>::type
	{
		return TO(static_cast<radiance>(c));
	}

	template <class DUMMY=int>
	static auto map(const radiance &c)
		-> typename std::enable_if<sizeof(DUMMY) &&
			std::is_arithmetic<TO>::value, TO>::type
	{
		// converts to luminance
		return math::map<TO>(0.2989*c.x + 0.5866*c.y + 0.1144*c.z);
	}

	template <class DUMMY=int>
	static auto map(const FROM &c)
		-> typename std::enable_if<sizeof(DUMMY) && 
						std::is_integral<FROM>::value &&
						std::is_integral<TO>::value &&
						(sizeof(TO) >= sizeof(FROM)), TO>::type
	{
		return c << (sizeof(TO)-sizeof(FROM))*8;
	}

	template <class DUMMY=int>
	static auto map(const FROM &c)
		-> typename std::enable_if<sizeof(DUMMY) && 
						std::is_integral<FROM>::value &&
						std::is_integral<TO>::value &&
						(sizeof(TO) < sizeof(FROM)), TO>::type
	{
		return c >> (sizeof(FROM)-sizeof(TO))*8;
	}

	template <class DUMMY=int>
	static auto map(const FROM &c)
		-> typename std::enable_if<sizeof(DUMMY) &&
			(std::is_floating_point<FROM>::value && 
					std::is_arithmetic<TO>::value) ||
			(std::is_arithmetic<FROM>::value && 
					std::is_floating_point<TO>::value),
				TO>::type
	{
		return math::map<TO>(math::unmap(c));
	}
};

template <class T>
struct cast_def<T, T>
{
	static const T &map(const T &c)
	{
		return c;
	}
};

// needed for disambiguation
template <template <class> class C, class T>
struct cast_def<C<T>, C<T>>
{
	static const C<T> &map(const C<T> &c)
	{
		return c;
	}
};

template <template<class> class C, class FROM, class TO>
struct cast_def<C<FROM>,C<TO>>
{
	template <class DUMMY=int>
	static auto map(const C<FROM> &c)
		-> typename std::enable_if<sizeof(DUMMY) && 
						std::is_integral<FROM>::value &&
						std::is_integral<TO>::value &&
						(sizeof(TO) >= sizeof(FROM)), C<TO>>::type
	{
		C<TO> o;
		for(size_t i=0; i<c.size(); ++i)
			o[i] = c[i] << (sizeof(TO)-sizeof(FROM))*8;
		return o;
	}

	template <class DUMMY=int>
	static auto map(const C<FROM> &c)
		-> typename std::enable_if<sizeof(DUMMY) && 
						std::is_integral<FROM>::value &&
						std::is_integral<TO>::value &&
						(sizeof(TO) < sizeof(FROM)), C<TO>>::type
	{
		C<TO> o;
		for(size_t i=0; i<c.size(); ++i)
			o[i] = c[i] >> (sizeof(FROM)-sizeof(TO))*8;
		return o;
	}

	template <class DUMMY=int>
	static auto map(const C<FROM> &c)
		-> typename std::enable_if<sizeof(DUMMY) && 
						(!std::is_integral<FROM>::value ||
						 !std::is_integral<TO>::value), C<TO>>::type
	{
		C<TO> o;
		for(size_t i=0; i<c.size(); ++i)
			o[i] = math::map<TO>(math::unmap(c[i]));
		return o;
	}
};

}} // namespace s3d::color

#endif
