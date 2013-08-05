#ifndef S3D_IMAGE_PARAMS_H
#define S3D_IMAGE_PARAMS_H

#include <unordered_map>
#include <string>
#include "../util/any.h"

namespace s3d { namespace img
{

template <class T=void> class named_parameter;

template <>
class named_parameter<>/*{{{*/
{
public:
	explicit named_parameter(const char *n) : m_name(n) {}

	const char *name() const { return m_name; }

protected:
	const char *m_name;
};/*}}}*/

template <class T>
class named_parameter : public named_parameter<>/*{{{*/
{
public:
	explicit named_parameter(const char *n) : named_parameter<>(n) {}

	std::pair<const char *,any> operator=(const T &val) const
		{ return {name(), val}; }
};/*}}}*/

class parameters/*{{{*/
{
	typedef std::unordered_map<std::string,any> container;
	container m_data;
public:
	parameters() {}

	template <class...PARAMS>
	parameters(const container::value_type &par, const PARAMS &...params)
	{
		insert(par, params...);
	}

	template <class...PARAMS>
	parameters(const parameters &par, const PARAMS &...params)
	{
		insert(par, params...);
	}

	parameters(parameters &&that)
		: m_data(std::move(that.m_data))
	{
	}

	parameters &operator=(const parameters &that) = default;
	parameters &operator=(parameters &&that)
	{
		m_data = std::move(that.m_data);
		return *this;
	}

	const any &operator[](const named_parameter<> &param) const
	{
		static any null_any;
		auto it = m_data.find(param.name());
		if(it != m_data.end())
			return it->second;
		else
			return null_any;
	}

	void insert() {}

	template <class...PARAMS>
	void insert(const container::value_type &par, const PARAMS &...params)
	{
		m_data.insert(par);
		insert(params...);
	}

	template <class...PARAMS>
	void insert(const parameters &par, const PARAMS &...params)
	{ 
		m_data.insert(par.m_data.begin(), par.m_data.end()); 
		insert(params...);
	}
};/*}}}*/

#define S3D_FORMAT_PARAMETER(name,T) \
	class name##_type : public named_parameter<T> \
	{ \
	public: \
		name##_type() : named_parameter<T>(#name) {} \
		using named_parameter<T>::operator=;\
	}; \
	static const name##_type name;

S3D_FORMAT_PARAMETER(background_color, color::rgb);
S3D_FORMAT_PARAMETER(scale, float);
S3D_FORMAT_PARAMETER(grayscale, bool);
S3D_FORMAT_PARAMETER(bpp, int);
S3D_FORMAT_PARAMETER(alpha, bool);
S3D_FORMAT_PARAMETER(quality, float); // 0% to 100%

}} // namespace s3d::img

#endif
