/*
	Copyright (c) 2009, Rodolfo Schulz de Lima.

	This file is part of S3D++.

	S3D++ is free software: you can redistribute it and/or modify 
	it under the terms of the GNU Lesser General Public License 
	version 3 as published by the Free Software Foundation.

	S3D++ is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public 
	License along with S3D++. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef S3D_UTIL_ANY_HPP
#define S3D_UTIL_ANY_HPP

#include <typeinfo>
#include <boost/lexical_cast.hpp>
#include <boost/noncopyable.hpp>
#include "clonable.h"
#include "type_traits.h"
#include "pointer.h"
#include "optional.h"

namespace s3d
{

namespace detail/*{{{*/
{
	template <class T, class EN=void>
	struct stringizable
	{
		static const int value = false;
		std::string conv(const T &) const { throw std::bad_cast(); }
	};

	template <class T>
	struct stringizable
	<
		T,
		typename std::enable_if<has_ostream_operator<T>::value>::type
	>
	{
		static const int value = true;
		std::string conv(const T &v) const 
		{ 
			return boost::lexical_cast<std::string>(v);
		}
	};

	// This is the base conversion class, whose members are called by
	// any_cast
	template <class T>
	class type_conversion
	{
	public:
		/* Due to gcc bug #32534 the compiler doesn't instantiate this static
		   member when it's first used. So we will make it a pointer and
		   durint type_conversion's construction we'll assign it to a
		   locally defined static vector so that we can control when
		   'conversions' will be properly instantiated.
		*/
		static std::vector<std::unique_ptr<type_conversion<T>>> *conversions;
		type_conversion()
		{
			static std::vector<std::unique_ptr<type_conversion<T>>> conv;
			conversions = &conv;
		}

		virtual bool can_convert(const any &a) const = 0;
		virtual T convert(const any &a) const = 0;
		virtual T convert(any &&a) const = 0;
	};
	template <class T> 
	std::vector<std::unique_ptr<type_conversion<T>>> *
		type_conversion<T>::conversions;

	template <class T, class F, class C=void>
	class TypeConversion : public type_conversion<T>
	{
	public:
		TypeConversion(C conv) : m_conv(conv) {}

		virtual bool can_convert(const any &a) const
		{
			return a.is_convertible_to<F>();
		}
		virtual T convert(const any &a) const
		{
			return m_conv(any_cast<F>(a));
		}
		virtual T convert(any &&a) const
		{
			return m_conv(any_cast<F>(std::move(a)));
		}
	private:
		C m_conv;
	};

	template <class T, class F>
	class TypeConversion<T,F,void> : public type_conversion<T>
	{
	public:
		static void create()
		{
			type_conversion<T>::conversions->push_back(make_unique<TypeConversion<T,F>>());
		}
		template <class C>
		static void create(C conv)
		{
			type_conversion<T>::conversions->push_back(make_unique<TypeConversion<T,F,C>>(conv));
		}

		virtual bool can_convert(const any &a) const
		{
			return a.is_convertible_to<F>();
		}
		virtual T convert(const any &a) const
		{
			return T(any_cast<F>(a));
		}
		virtual T convert(any &&a) const
		{
			return T(any_cast<F>(std::move(a)));
		}
	};

	template <class T> 
	bool can_convert_to(const any &a)
	{
		if(type_conversion<T>::conversions == NULL)
			return false;

		for(unsigned i=0; i<type_conversion<T>::conversions->size(); ++i)
		{
			if((*type_conversion<T>::conversions)[i]->can_convert(a))
				return true;
		}
		return false;
	}

	template <class T, class A> 
	T convert_to(A &&a)
	{
		if(type_conversion<T>::conversions != NULL)
		{
			for(unsigned i=0; i<type_conversion<T>::conversions->size(); ++i)
			{
				if((*type_conversion<T>::conversions)[i]->can_convert(a))
					return (*type_conversion<T>::conversions)[i]->convert(std::forward<A>(a));
			}
		}
		throw std::bad_cast();
	}
}/*}}}*/

// Classes that transform the held object into a common representation

class any::holder : public clonable/*{{{*/
{
public:
	holder(const holder &that) : m_type(that.m_type) {}
	holder(const std::type_info &type) : m_type(type) {}
	virtual ~holder() {}

	virtual bool can_cast_as_floating_point() const { return false; }
	virtual double cast_as_floating_point() const { throw std::bad_cast(); }

	virtual bool can_cast_as_string() const { return false; }
	virtual std::string cast_as_string() const { throw std::bad_cast(); }

	virtual bool can_cast_as_integer() const { return false; }
	virtual long long cast_as_integer() const { throw std::bad_cast(); }

	virtual bool can_cast_as_conv_from_any_ptr() const { return false; }
	virtual convertible_from_any * 
		cast_as_conv_from_any_ptr() const { throw std::bad_cast(); }

	virtual bool can_cast_as_conv_from_any_shared() const { return false; }
	virtual std::shared_ptr<convertible_from_any>
		cast_as_conv_from_any_shared() const { throw std::bad_cast(); }

	virtual std::shared_ptr<convertible_from_any>
		cast_as_conv_from_any_shared() 
	{ 
		throw const_cast<const holder *>(this)->cast_as_conv_from_any_shared();
	}

	virtual bool can_cast_as_conv_from_any_unique() const { return false; }
	virtual std::unique_ptr<convertible_from_any>
		cast_as_conv_from_any_unique() { throw std::bad_cast(); }

	const std::type_info &type() const { return m_type; }

	DEFINE_PURE_CLONABLE(holder);
private:
	const std::type_info &m_type;
};/*}}}*/

template <class T, class EN>
class any::Holder /*{{{ : other types */
	: public holder, detail::stringizable<T>
{
public:
	static_assert(!std::is_convertible<T *, const any *>::value,
				"Logic error, cannot create a holder to an 'any'");

	Holder(Holder &&that) 
		: holder(that), m_value(std::move(that.m_value)) {}
	template <class U> Holder(U &&v) 
		: holder(typeid(T)), m_value(std::forward<U>(v)) {}

	virtual bool can_cast_as_string() const 
	{ 
		return detail::stringizable<T>::value;
	}
	virtual std::string cast_as_string() const 
	{ 
		return detail::stringizable<T>::conv(m_value);
	}

	T m_value;

	DEFINE_CLONABLE(Holder);
};/*}}}*/

template <class T>
class any::Holder /*{{{ : noncopyable types */
<
	T,
	typename std::enable_if
	<
		std::is_base_of<boost::noncopyable, T>::value
	>::type
>
	: public holder, detail::stringizable<T>
{
public:
	Holder(Holder &&that) 
		: holder(that), m_value(std::move(that.m_value)) {}
	template <class U> Holder(U &&v) 
		: holder(typeid(T)), m_value(std::forward<U>(v)) {}

	virtual bool can_cast_as_string() const 
	{ 
		return detail::stringizable<T>::value;
	}
	virtual std::string cast_as_string() const 
	{ 
		return detail::stringizable<T>::conv(m_value);
	}

	T m_value;

	virtual Holder *do_clone() const 
		{ throw std::runtime_error("Cannot clone non CopyConstructible objects"); }
};/*}}}*/

template <class T, class D>
class any::Holder /*{{{ : unique_ptr */
<
	std::unique_ptr<T,D>,
	typename std::enable_if
	<
		!std::is_base_of<convertible_from_any, T>::value
	>::type
>
	: public holder, detail::stringizable<T>
{
public:
	Holder(Holder &&that) 
		: holder(that), m_value(std::move(that.m_value)) {}
	Holder(std::unique_ptr<T,D> &&v) 
		: holder(typeid(std::unique_ptr<T>)), m_value(std::move(v)) {}

	virtual bool can_cast_as_string() const 
	{ 
		return detail::stringizable<T>::value;
	}
	virtual std::string cast_as_string() const 
	{ 
		if(m_value)
			return detail::stringizable<T>::conv(*m_value);
		else
			return "(NULL";
	}

	std::unique_ptr<T,D> m_value;

private:
	virtual Holder *do_clone() const 
		{ throw std::runtime_error("Cannot clone non CopyConstructible objects"); }
};/*}}}*/

template <class T>
class any::Holder /*{{{ : convertible_from_any pointer */
<
	T *,
	typename std::enable_if
	<
		std::is_base_of<convertible_from_any, T>::value
	>::type
>
	: public holder, detail::stringizable<T>
{
public:
	Holder(Holder &&that) 
		: holder(that), m_value(std::move(that.m_value)) {}
	template <class U> Holder(U &&v) 
		: holder(typeid(T *)), m_value(std::forward<U>(v)) {}

	virtual bool can_cast_as_conv_from_any_ptr() const { return true; }
	virtual convertible_from_any * cast_as_conv_from_any_ptr() const 
	{ 
		return static_cast<convertible_from_any *>(m_value);
	}

	virtual bool can_cast_as_string() const 
	{ 
		return detail::stringizable<T>::value;
	}
	virtual std::string cast_as_string() const 
	{ 
		if(m_value)
			return detail::stringizable<T>::conv(*m_value);
		else
			return "(NULL";
	}

	T *m_value;

	DEFINE_CLONABLE(Holder);
};/*}}}*/

template <class T>
class any::Holder /*{{{ : convertible_from_any reference */
<
	T &,
	typename std::enable_if
	<
		std::is_base_of< convertible_from_any, T>::value
	>::type
>
	: public holder, detail::stringizable<T>
{
public:
	Holder(Holder &&that) 
		: holder(that), m_value(std::move(that.m_value)) {}
	template <class U> Holder(U &&v) 
		: holder(typeid(T &)), m_value(std::forward<U>(v)) {}

	virtual bool can_cast_as_conv_from_any_ptr() const { return true; }
	virtual convertible_from_any * cast_as_conv_from_any_ptr() const 
	{ 
		return static_cast<convertible_from_any *>(&m_value);
	}

	virtual bool can_cast_as_string() const 
	{ 
		return detail::stringizable<T>::value;
	}
	virtual std::string cast_as_string() const 
	{ 
		return detail::stringizable<T>::conv(m_value);
	}

	T &m_value;

	DEFINE_CLONABLE(Holder);
};/*}}}*/

template <class T>
class any::Holder /*{{{ : convertible_from_any shared_ptr */
<
	std::shared_ptr<T>,
	typename std::enable_if
	<
		std::is_base_of<convertible_from_any, T>::value
	>::type
>
	: public holder, detail::stringizable<T>
{
public:
	Holder(Holder &&that) 
		: holder(that), m_value(std::move(that.m_value)) {}
	template <class U> Holder(U &&v) 
		: holder(typeid(std::shared_ptr<T>)), m_value(std::forward<U>(v)) {}

	virtual bool can_cast_as_conv_from_any_shared() const { return true; }
	virtual std::shared_ptr<convertible_from_any> 
		cast_as_conv_from_any_shared() const 
	{ 
		return std::static_pointer_cast<convertible_from_any>(m_value); 
	}

	virtual std::shared_ptr<convertible_from_any> 
		cast_as_conv_from_any_shared()
	{ 
		std::shared_ptr<convertible_from_any> ret = std::move(m_value);
		return std::static_pointer_cast<convertible_from_any>(ret); 
	}

	virtual bool can_cast_as_conv_from_any_ptr() const { return true; }
	virtual convertible_from_any * 
		cast_as_conv_from_any_ptr() const { return m_value.get(); }

	virtual bool can_cast_as_string() const 
	{ 
		return detail::stringizable<T>::value;
	}
	virtual std::string cast_as_string() const 
	{ 
		if(m_value)
			return detail::stringizable<T>::conv(*m_value);
		else
			return "(NULL)";
	}

	std::shared_ptr<T> m_value;

	DEFINE_CLONABLE(Holder);
};/*}}}*/

template <class T, class D>
class any::Holder /*{{{ : convertible_from_any unique_ptr */
<
	std::unique_ptr<T,D>,
	typename std::enable_if
	<
		std::is_base_of<convertible_from_any, T>::value
	>::type
>
	: public holder, detail::stringizable<T>
{
public:
	Holder(Holder &&that) 
		: holder(that), m_value(std::move(that.m_value)) {}
	Holder(std::unique_ptr<T,D> &&v) 
		: holder(typeid(std::unique_ptr<T>)), m_value(std::move(v)) {}

	virtual bool can_cast_as_conv_from_any_unique() const { return true; }
	virtual std::unique_ptr<convertible_from_any>
		cast_as_conv_from_any_unique() 
	{ 
		return std::move(m_value); 
	}

	virtual bool can_cast_as_conv_from_any_shared() const { return true; }
	virtual std::shared_ptr<convertible_from_any>
		cast_as_conv_from_any_shared() 
	{ 
		return std::shared_ptr<convertible_from_any>(m_value.release());
	}

	virtual bool can_cast_as_conv_from_any_ptr() const { return true; }
	virtual convertible_from_any * 
		cast_as_conv_from_any_ptr() const { return m_value.get(); }

	virtual bool can_cast_as_string() const 
	{ 
		return detail::stringizable<T>::value;
	}
	virtual std::string cast_as_string() const 
	{ 
		if(m_value)
			return detail::stringizable<T>::conv(*m_value);
		else
			return "(NULL)";
	}

	std::unique_ptr<T,D> m_value;

private:
	virtual Holder *do_clone() const 
		{ throw std::runtime_error("Cannot clone non CopyConstructible objects"); }
};/*}}}*/

template <class T>
class any::Holder/*{{{ : arithmetic types */
<
	T,
	typename std::enable_if
	<
		std::is_arithmetic<T>::value
	>::type
>
	: public holder
{
public:
	Holder(Holder &&that) 
		: holder(that), m_value(std::move(that.m_value)) {}
	template <class U> Holder(U &&v) 
		: holder(typeid(T)), m_value(std::forward<U>(v)) {}

	virtual bool can_cast_as_string() const
	{
		try
		{
			cast_as_string();
			return true;
		}
		catch(std::bad_cast &)
		{
			return false;
		}
	}
	virtual std::string cast_as_string() const
	{
		return boost::lexical_cast<std::string>(m_value);
	}
	virtual bool can_cast_as_floating_point() const { return true; }
	virtual double cast_as_floating_point() const
	{
		return m_value;
	}
	virtual bool can_cast_as_integer() const { return true; }
	virtual long long cast_as_integer() const
	{
		return m_value;
	}
	T m_value;

	DEFINE_CLONABLE(Holder);
};/*}}}*/

template <>
class any::Holder<std::string>/*{{{ : strings */
	: public holder
{
public:
	Holder(Holder &&that) 
		: holder(that), m_value(std::move(that.m_value)) {}
	template <class U> Holder(U &&v) 
		: holder(typeid(std::string)), m_value(std::forward<U>(v)) {}

	virtual bool can_cast_as_string() const { return true; }
	virtual std::string cast_as_string() const
	{
		return m_value;
	}

	virtual bool can_cast_as_floating_point() const
	{
		try
		{
			cast_as_floating_point();
			return true;
		}
		catch(std::bad_cast &)
		{
			return false;
		}
	}
	
	virtual double cast_as_floating_point() const
	{
		return boost::lexical_cast<double>(m_value);
	}

	virtual bool can_cast_as_integer() const
	{
		try
		{
			cast_as_floating_point();
			return true;
		}
		catch(std::bad_cast &)
		{
			return false;
		}
	}

	virtual long long cast_as_integer() const
	{
		return boost::lexical_cast<long long>(m_value);
	}

	std::string m_value;

	DEFINE_CLONABLE(Holder);
};/*}}}*/

// Classes that transform the common representation into the destination type

template <class U, class EN>
struct any::Cast/*{{{ : other types*/
{
	bool can_cast(const any::holder &v) const 
	{ 
		return false; 
	}
	U operator()(const any::holder &v) const
	{
		throw std::bad_cast();
	}
};/*}}}*/

template <class U>
struct any::Cast/*{{{ : integral types*/
<
	U, 
	typename std::enable_if
	<
		std::is_integral<U>::value
	>::type
>
{
	bool can_cast(const any::holder &v) const 
	{ 
		return v.can_cast_as_integer(); 
	}
	U operator()(const any::holder &v) const
	{
		return v.cast_as_integer();
	}
};/*}}}*/

template <class U>
struct any::Cast/*{{{ : floating point types*/
<
	U, 
	typename std::enable_if
	<
		std::is_floating_point<U>::value
	>::type
>
{
	bool can_cast(const any::holder &v) const 
	{ 
		return v.can_cast_as_floating_point(); 
	}
	U operator()(const any::holder &v) const
	{
		return v.cast_as_floating_point();
	}
};/*}}}*/

template <class U>
struct any::Cast/*{{{ : convertible_from_any ptr*/
<
	U *, 
	typename std::enable_if
	<
		std::is_base_of<convertible_from_any, U>::value
	>::type
>
{
	bool can_cast(const any::holder &v) const 
	{ 
		return v.can_cast_as_conv_from_any_ptr() &&
			   dynamic_cast<U *>(v.cast_as_conv_from_any_ptr());
	}
	U * operator()(const any::holder &v) const
	{
		return &dynamic_cast<U &>(*v.cast_as_conv_from_any_ptr());
	}
};/*}}}*/

template <class U>
struct any::Cast/*{{{ : convertible_from_any ref*/
<
	U &, 
	typename std::enable_if
	<
		std::is_base_of<convertible_from_any, U>::value
	>::type
>
{
	bool can_cast(const any::holder &v) const 
	{ 
		return v.can_cast_as_conv_from_any_ptr() &&
			   dynamic_cast<U *>(v.cast_as_conv_from_any_ptr());
	}
	U& operator()(const any::holder &v) const
	{
		return dynamic_cast<U &>(*v.cast_as_conv_from_any_ptr());
	}
};/*}}}*/

template <class U>
struct any::Cast/*{{{ : convertible_from_any value*/
<
	U, 
	typename std::enable_if
	<
		std::is_base_of<convertible_from_any, U>::value
	>::type
>
{
	bool can_cast(const any::holder &v) const 
	{ 
		return v.can_cast_as_conv_from_any_ptr() &&
			   dynamic_cast<U *>(v.cast_as_conv_from_any_ptr());
	}
	U operator()(const any::holder &v) const
	{
		return U(dynamic_cast<const U &>(*v.cast_as_conv_from_any_ptr()));
	}

	U operator()(any::holder &&v) const
	{
		return U(std::move(dynamic_cast<U &>(*v.cast_as_conv_from_any_ptr())));
	}
};/*}}}*/

template <class U>
struct any::Cast/*{{{ : convertible_from_any shared_ptr*/
<
	std::shared_ptr<U>, 
	typename std::enable_if
	<
		std::is_base_of<convertible_from_any, U>::value
	>::type
>
{
	bool can_cast(const any::holder &v) const 
	{ 
		return (v.can_cast_as_conv_from_any_shared() ||
			   v.can_cast_as_conv_from_any_unique()) &&
			   dynamic_cast<U *>(v.cast_as_conv_from_any_ptr());

	}
	std::shared_ptr<U> operator()(const any::holder &v) const
	{
		return std::dynamic_pointer_cast<U>(v.cast_as_conv_from_any_shared());
	}
	std::shared_ptr<U> operator()(any::holder &&v) const
	{
		if(v.can_cast_as_conv_from_any_shared())
			return std::dynamic_pointer_cast<U>(v.cast_as_conv_from_any_shared());
		else
		{
			dynamic_cast<U &>(*v.cast_as_conv_from_any_ptr());
			return std::dynamic_pointer_cast<U>(make_shared(v.cast_as_conv_from_any_unique()));
		}
	}
};/*}}}*/

template <class U>
struct any::Cast/*{{{ : convertible_from_any unique_ptr*/
<
	std::unique_ptr<U>, 
	typename std::enable_if
	<
		std::is_base_of<convertible_from_any, U>::value
	>::type
>
{
	template <class V>
	std::unique_ptr<U> operator()(V &v) const
	{
		throw std::bad_cast();
	}

	bool can_cast(const any::holder &v) const
	{
		return v.can_cast_as_conv_from_any_unique() &&
			   dynamic_cast<U *>(v.cast_as_conv_from_any_ptr());
	}

	std::unique_ptr<U> operator()(any::holder &&v) const
	{
		dynamic_cast<U &>(*v.cast_as_conv_from_any_ptr());
		return std::unique_ptr<U>(
			static_cast<U*>(v.cast_as_conv_from_any_unique().release()));
	}
};/*}}}*/

template <class U>
struct any::Cast/*{{{ : convertible_from_any auto_ptr*/
<
	std::auto_ptr<U>, 
	typename std::enable_if
	<
		std::is_base_of<convertible_from_any, U>::value
	>::type
>
{
	template <class V>
	std::auto_ptr<U> operator()(V &v) const
	{
		throw std::bad_cast();
	}

	bool can_cast(const any::holder &v) const
	{
		return v.can_cast_as_conv_from_any_unique() &&
			   dynamic_cast<U *>(v.cast_as_conv_from_any_ptr());
	}

	std::auto_ptr<U> operator()(any::holder &&v) const
	{
		dynamic_cast<U &>(*v.cast_as_conv_from_any_ptr());
		return std::auto_ptr<U>(
			static_cast<U*>(v.cast_as_conv_from_any_unique().release()));
	}
};/*}}}*/

template <>
struct any::Cast<std::string>/*{{{ : strings*/
{
	bool can_cast(const any::holder &v) const
	{
		return v.can_cast_as_string();
	}
	std::string operator()(const any::holder &v) const
	{
		return v.cast_as_string();
	}
};/*}}}*/

template <class U>
struct any::Cast<optional<U>>/*{{{ : optional */
{
	bool can_cast(const any::holder &v) const
	{
		return v.type() == typeid(typename remove_cv_ref<U>::type);
	}
	optional<U> operator()(const any::holder &v) const
	{
		if(!can_cast(v))
			throw std::bad_cast();
		return static_cast<const any::Holder<typename remove_cv_ref<U>::type > &>(v).m_value;
	}

	optional<U> operator()(any::holder &&v) const
	{
		if(!can_cast(v))
			throw std::bad_cast();
		return std::move(static_cast<any::Holder<typename remove_cv_ref<U>::type > &>(v).m_value);
	}
};/*}}}*/

template <class T> 
any::any(T &&v)/*{{{*/
	: m_holder(new Holder<typename remove_cv_ref<T>::type>(std::forward<T>(v)))
{
}/*}}}*/

inline any::any(const any &that)/*{{{*/
	: m_holder(that.m_holder ? that.m_holder->clone().release() : NULL)
{
}/*}}}*/
inline any::any(any &that)/*{{{*/
	: m_holder(that.m_holder ? that.m_holder->clone().release() : NULL)
{
}/*}}}*/
inline any::any(any &&that)/*{{{*/
	: m_holder(std::move(that.m_holder))
{
	that.m_holder = NULL;
}/*}}}*/
inline any::~any()/*{{{*/
{
	delete m_holder;
	m_holder = NULL; // só pra garantir
}/*}}}*/

template <class T> 
any &any::operator=(T &&v)/*{{{*/
{
	delete m_holder;
	m_holder = new Holder<typename remove_cv_ref<T>::type>(std::forward<T>(v));
	return *this;
}/*}}}*/

inline any &any::operator=(const any &that)/*{{{*/
{
	if(that.m_holder)
	{
		delete m_holder;
		m_holder = that.m_holder->clone().release();
	}
	else
	{
		delete m_holder;
		m_holder = NULL;
	}
		
	return *this;
}/*}}}*/
inline any &any::operator=(any &that)/*{{{*/
{
	return *this = const_cast<const any &>(that);
}/*}}}*/
inline any &any::operator=(any &&that)/*{{{*/
{
	delete m_holder;
	m_holder = std::move(that.m_holder);
	that.m_holder = NULL;
	return *this;
}/*}}}*/

inline const std::type_info &any::type() const /*{{{*/
{ 
	if(m_holder)
		return m_holder->type();
	else
		return typeid(void);
}/*}}}*/

template <class T> 
T any_cast(const any &a)/*{{{*/
{
	if(!a)
	{
		throw bad_any_cast(format("Cannot convert empty value to type '%s'",
									  demangle(typeid(T).name())));
	}

	// Can we do a direct cast?
	if(a.m_holder->type() == typeid(typename remove_cv_ref<T>::type))
		return static_cast<any::Holder<typename remove_cv_ref<T>::type > &>(*a.m_holder).m_value;

	try
	{
		if(detail::can_convert_to<T>(a))
			return detail::convert_to<T>(a);

		return any::Cast<T>()(*a.m_holder);
	}
	catch(std::bad_cast &)
	{
		throw bad_any_cast(format("Cannot convert value from type '%s' to '%s'",
						   demangle(a.type().name()),
						   demangle(typeid(T).name())));
	}
}/*}}}*/

template <class T> 
T any_cast(any &&a)/*{{{*/
{
	if(!a)
	{
		throw bad_any_cast(format("Cannot convert empty value to type '%s'",
									  demangle(typeid(T).name())));
	}

	// Can we do a direct cast?
	if(a.m_holder->type() == typeid(typename remove_cv_ref<T>::type))
	{
		T aux = std::move(static_cast<any::Holder<typename remove_cv_ref<T>::type> &>(*a.m_holder).m_value);
		a = any();
		return std::move(aux);
	}

	try
	{
		if(detail::can_convert_to<T>(a))
			return detail::convert_to<T>(std::move(a));

		T aux = any::Cast<T>()(std::move(*a.m_holder));
		a = any();
		return std::move(aux);
	}
	catch(std::bad_cast &)
	{
		throw bad_any_cast(format("Cannot convert value from type '%s' to '%s'",
						   demangle(a.type().name()),
						   demangle(typeid(T).name())));
	}
}/*}}}*/

template <> 
inline any any_cast<any>(const any &a)/*{{{*/
{
	return a;
}/*}}}*/

template <> 
inline any any_cast<any>(any &&a)/*{{{*/
{
	return std::move(a);
}/*}}}*/

template <class T> 
bool any::is_convertible_to() const/*{{{*/
{
	if(!*this)
		return false;

	if(m_holder->type() == typeid(typename remove_cv_ref<T>::type))
		return true;

	return Cast<T>().can_cast(*m_holder);
}/*}}}*/

template <>
inline bool any::is_convertible_to<any>() const /*{{{*/
{ 
	return true; 
}/*}}}*/


/*
   Here's the problem: we have to be able to convert between unrelated
   types. I don't know a way to specify this conversion using the
   type's implicit conversion functions (ctor, operator T,...) because
   we cannot know at compile time which conversions we'll be using.

   BUT... the programmer can specify at compile time which conversions 
   will be valid. This is done by 'define_type_conversion' struct like
   this:

   define_type_conversion<T, F> conv1;
   define_type_conversion<T, F> conv2(some_functor);

   This can be done anywhere, preferably one per type combination
   (conv1/conv2 can be a static global/local variable).
   conv1 defines a conversion from 'F' to 'T' using T's constructor from
   'F'. conv2 specifies that the conversion from 'F' to 'T' is done
   by function 'some_functor'.
*/

template <class T, class F>
struct define_type_conversion
{
	template <class... C>
	define_type_conversion(C &&...conv)
	{
		detail::TypeConversion<T,F>::create(conv...);
	}
};

} // namespace s3d

#endif

// $Id: any.hpp 2888 2010-07-27 23:29:05Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

