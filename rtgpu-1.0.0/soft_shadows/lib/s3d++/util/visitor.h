#ifndef S3D_UTIL_VISITOR_H
#define S3D_UTIL_VISITOR_H

#include "recursion_guard.h"
#include "../mpl/vector_fwd.h"

namespace s3d
{

// Visitor design pattern

class visitor_base/*{{{*/
{
public:
    virtual ~visitor_base() {}

    template <class T> 
	auto visit(T &visitable)
		-> typename std::enable_if<!std::is_const<T>::value,bool>::type;

    template <class T> 
	auto visit(T &&visitable)
		-> typename std::enable_if<!std::is_const<T>::value &&
	                               !std::is_reference<T>::value, bool>::type;

    template <class T> 
	bool visit(const T &visitable);
};/*}}}*/

template <class T> class visitor_def;
template <class T> class visitor_def_impl;

template <class T> 
class visitor_def<T &>/*{{{*/
{
public:
    bool visit(T &visitable) { return do_visit(visitable); }
private:

    virtual bool do_visit(T &visitable) = 0;
};/*}}}*/
template <class T> 
class visitor_def_impl<T&> : public visitor_def<T&>/*{{{*/
{
public:
	template <class IMPL>
	visitor_def_impl(const IMPL &impl) 
		 : m_impl(impl) {}
private:
	std::function<bool(T &)> m_impl;

    virtual bool do_visit(T &visitable)
	{
		assert(m_impl);
		return m_impl(visitable);
	}
};/*}}}*/

template <class T> 
class visitor_def<T&&>/*{{{*/
{
public:
    bool visit(T &&visitable) { return do_visit(std::move(visitable)); }
private:

    virtual bool do_visit(T &&visitable) = 0;
};/*}}}*/
template <class T> 
class visitor_def_impl<T&&> : public visitor_def<T&&>/*{{{*/
{
public:
	template <class IMPL>
	visitor_def_impl(const IMPL &impl) : m_impl(impl) {}
private:
	std::function<bool(T &&)> m_impl;

    virtual bool do_visit(T &&visitable)
	{
		assert(m_impl);
		return m_impl(std::move(visitable));
	}
};/*}}}*/

template <class T> 
class visitor_def : visitor_def<T &>/*{{{*/
{
};/*}}}*/
template <class T> 
class visitor_def_impl : public visitor_def_impl<T &>/*{{{*/
{
public:
	template <class IMPL>
	visitor_def_impl(const IMPL &impl) : visitor_def_impl<T&>(impl) {}
};/*}}}*/

template <class... TYPES>
class visitor/*{{{*/
    : public virtual visitor_base
	, public visitor_def<TYPES>...
{
public:
	typedef visitor visitor_type;
};/*}}}*/
template <class... TYPES> 
class const_visitor : public visitor<const TYPES &...>/*{{{*/
{
public:
	typedef const_visitor visitor_type;
};/*}}}*/
template <class... TYPES> 
class move_visitor : public visitor<TYPES &&...>/*{{{*/
{
public:
	typedef move_visitor visitor_type;
};/*}}}*/

template <class... TYPES>
class visitor_impl/*{{{*/
	: public virtual visitor_base
	, public visitor_def_impl<TYPES>...
{
public:
	typedef visitor_impl visitor_type;

	template <class IMPL>
	visitor_impl(IMPL impl) 
		: visitor_def_impl<TYPES>(impl)... {}
};/*}}}*/
template <class... TYPES> 
class const_visitor_impl : public visitor_impl<const TYPES &...>/*{{{*/
{
public:
	typedef const_visitor_impl visitor_type;

	template <class IMPL>
	const_visitor_impl(IMPL impl) : visitor_impl<const TYPES &...>(impl) {}
};/*}}}*/
template <class... TYPES> 
class move_visitor_impl : public visitor_impl<TYPES &&...>/*{{{*/
{
public:
	typedef move_visitor_impl visitor_type;

	template <class IMPL>
	move_visitor_impl(IMPL impl) : visitor_impl<TYPES &&...>(impl) {}
};/*}}}*/

template <class... TYPES>
class visitor<mpl::vector<TYPES...>>/*{{{*/
    : public virtual visitor_base
	, public visitor_def<TYPES>...
{
public:
	typedef visitor visitor_type;
};/*}}}*/
template <class... TYPES> 
class const_visitor<mpl::vector<TYPES...>> : public visitor<const TYPES &...>/*{{{*/
{
public:
	typedef const_visitor visitor_type;
};/*}}}*/
template <class... TYPES> 
class move_visitor<mpl::vector<TYPES...>> : public visitor<TYPES &&...>/*{{{*/
{
public:
	typedef move_visitor visitor_type;
};/*}}}*/

template <class... TYPES>
class visitor_impl<mpl::vector<TYPES...>>/*{{{*/
	: public virtual visitor_base
	, public visitor_def_impl<TYPES>...
{
public:
	typedef visitor_impl visitor_type;

	template <class IMPL>
	visitor_impl(IMPL impl) 
		: visitor_def_impl<TYPES>(impl)... {}
};/*}}}*/
template <class... TYPES> 
class const_visitor_impl<mpl::vector<TYPES...>> : public visitor_impl<const TYPES &...>/*{{{*/
{
public:
	typedef const_visitor_impl visitor_type;

	template <class IMPL>
	const_visitor_impl(IMPL impl) : visitor_impl<const TYPES &...>(impl) {}
};/*}}}*/
template <class... TYPES> 
class move_visitor_impl<mpl::vector<TYPES...>> : public visitor_impl<TYPES &&...>/*{{{*/
{
public:
	typedef move_visitor_impl visitor_type;

	template <class IMPL>
	move_visitor_impl(IMPL impl) : visitor_impl<TYPES &&...>(impl) {}
};/*}}}*/

template <class T> 
auto visitor_base::visit(T &visitable)/*{{{*/
	-> typename std::enable_if<!std::is_const<T>::value,bool>::type 
{
    // Podemos usar um visitable não const em um visitor não const ou const
    // (nesta ordem)

    if(auto *visitor = dynamic_cast<visitor_def<T &> *>(this))
        return visitor->visit(visitable);
    else if(auto *visitor = dynamic_cast<visitor_def<const T &> *>(this))
        return visitor->visit(visitable);
    else
        return false; // continua a visitação
}/*}}}*/
template <class T> 
auto visitor_base::visit(T &&visitable)/*{{{*/
	-> typename std::enable_if<!std::is_const<T>::value &&
	                           !std::is_reference<T>::value ,bool>::type 
{
    // Podemos usar um visitable não const em um visitor não const ou const
    // (nesta ordem)

    if(auto *visitor = dynamic_cast<visitor_def<T &&> *>(this))
        return visitor->visit(std::move(visitable));
    else
        return false; // continua a visitação
}/*}}}*/
template <class T> 
bool visitor_base::visit(const T &visitable)/*{{{*/
{
    // Só podemos usar um visitable const em um visitor const
    if(auto *visitor = dynamic_cast<visitor_def<const T &> *>(this))
        return visitor->visit(visitable);
    else
        return false; // continua a visitação
}/*}}}*/

class const_visitable
{
public:
    const_visitable() {}
    const_visitable(const const_visitable &) {}
    const_visitable &operator=(const const_visitable &) { return *this; }

    // visitor é um rvalue-reference pois ele pode ser um temporário
    bool accept(visitor_base &&visitor) const;

protected:
    mutable recursion_guard m_recguard;
private:
    virtual bool do_accept(visitor_base &&visitor) const = 0;

};

class visitable : public const_visitable
{
public:
    visitable() {}
    visitable(const visitable &that) : const_visitable(that) {}
    visitable &operator=(const visitable &that) 
        { const_visitable::operator=(that); return *this; }

    bool accept(visitor_base &&visitor);
    bool move_accept(visitor_base &&visitor);
    using const_visitable::accept;

private:
    virtual bool do_accept(visitor_base &&visitor) = 0;
    virtual bool do_move_accept(visitor_base &&visitor)
	{
		return do_accept(std::move(visitor));
	}
};

}

#endif
