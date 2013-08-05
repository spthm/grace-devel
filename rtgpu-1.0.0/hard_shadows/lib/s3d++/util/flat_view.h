#ifndef S3D_UTIL_RANGE_H
#define S3D_UTIL_RANGE_H

#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/iterator_range.hpp>
#include <boost/range/reference.hpp>
#include "type_traits.h"

namespace s3d
{

// TODO: This is utterly convoluted, but works. Try a better solution next time

template <class IT, int L>
class flat_iterator
	: public boost::iterator_facade
	  <
		flat_iterator<IT,L>,
		typename value_type<IT,L+1>::type,
//		typename boost::iterator_category<IT>::type,
		typename boost::bidirectional_traversal_tag,
		typename copy_const<typename boost::iterator_reference<IT>::type,
			typename value_type<IT,L+1>::type>::type &
	  >
{
	typedef typename flat_iterator::iterator_facade_ facade;

	typedef boost::iterator_range<IT> parent_range;
	typedef boost::iterator_range
	<
		flat_iterator
		<
			typename boost::range_iterator
			<
				typename std::remove_reference
				<
					typename boost::iterator_reference<IT>::type
				>::type
			>::type,
			L-1
		> 
	> children_range;

	typedef typename boost::range_iterator<children_range>::type child_iterator;
	typedef IT parent_iterator;

	static typename boost::iterator_value<IT>::type null_child;

public:
	typedef typename s3d::value_type<IT,L+1>::type value_type;
	typedef flat_iterator this_type;
	typedef flat_iterator type;

	flat_iterator() = default;

	template <class D=int>
	flat_iterator(IT itparent, parent_range parents)
		: m_parents(parents)
		, m_children(itparent == parents.end()
				 ? children_range(child_iterator(null_child.begin(),null_child),
								  child_iterator(null_child.end(), null_child))
				 : children_range(child_iterator(boost::begin(*itparent), 
												 *itparent),
								  child_iterator(boost::end(*itparent), 
												 *itparent)))
		, m_itparent(itparent)
		, m_itchild(m_children.begin())
	{
		rewind();
	}

	template <class IT2, class =
		typename std::enable_if<std::is_convertible<IT2,IT>::value>::type>
	flat_iterator(const flat_iterator<IT2,L> &that)
		: m_parents(that.m_parents)
		, m_children(that.m_children)
		, m_itparent(that.m_itparent)
		, m_itchild(that.m_itchild)
	{
	}

private:
	friend class boost::iterator_core_access;
	template <class IT2, int L2> friend class flat_iterator;

	void rewind()
	{
		// go to the first child
		while(m_itparent != m_parents.end() && empty(m_children))
			increment();
	}


	template <class IT2>
	bool equal(const flat_iterator<IT2,L> &that) const
	{
		assert(m_itchild  != that.m_itchild ||
			   m_itparent == that.m_itparent);

		return m_itchild == that.m_itchild;
	}

	void increment()
	{
		assert(m_itparent != m_parents.end());

		// m_itchild can be == m_children.end() during initial rewind()
		if(m_itchild != m_children.end())
			++m_itchild;

		while(m_itchild == m_children.end())
		{
			// are there more parents?
			if(++m_itparent != m_parents.end())
			{
				m_children = 
					 children_range(
						 child_iterator(boost::begin(*m_itparent), 
									    *m_itparent),
						 child_iterator(boost::end(*m_itparent), 
										*m_itparent));
				m_itchild = m_children.begin();
			}
			else
			{
				 m_children = children_range(
						 child_iterator(boost::begin(null_child), 
										null_child),
						 child_iterator(boost::end(null_child), 
										null_child));
				m_itchild = m_children.begin();
				break;
			}
		}
	}

	void decrement()
	{
		assert(m_itparent != m_parents.begin() || 
			   m_itchild != m_children.begin());

		if(m_itchild != m_children.begin())
			std::advance(m_itchild,-1);
		else
		{
			while(boost::empty(*--m_itparent))
				assert(m_itparent != m_parents.begin());

			m_children = 
				 children_range(
					 child_iterator(boost::begin(*m_itparent), 
									*m_itparent),
					 child_iterator(boost::end(*m_itparent), 
									*m_itparent));

			assert(!empty(m_children));
			m_itchild = m_children.end();
			std::advance(m_itchild,-1);
		}
	}

	void advance(int n)
	{
		if(n >= 0)
		{
			while(n > 0)
			{
				auto d = std::distance(m_itchild, m_children.end());

				if(d < n)
				{
					n -= d;
					assert(m_itparent != m_parents.end());
					std::advance(m_itparent,1);

					m_children = 
						 children_range(
							 child_iterator(boost::begin(*m_itparent), 
											*m_itparent),
							 child_iterator(boost::end(*m_itparent), 
											*m_itparent));
					m_itchild = m_children.begin();
				}
				else
				{
					n = 0;
					std::advance(m_itchild,n);
				}
			}
		}
		else
		{
			n = -n;
			while(n > 0)
			{
				auto d = std::distance(m_children.begin(), m_itchild);

				if(d < n)
				{
					n -= d;

					assert(m_itparent != m_parents.begin());
					std::advance(m_itparent,-1);

					m_children = 
						 children_range(
							 child_iterator(boost::begin(*m_itparent), 
											*m_itparent),
							 child_iterator(boost::end(*m_itparent), 
											*m_itparent));
					m_itchild = m_children.begin();
				}
				else
				{
					n = 0;
					std::advance(m_itchild, -n);
				}
			}
		}
	}

	auto distance_to(const flat_iterator &that) const
		-> typename facade::difference_type 
	{
		typedef typename facade::difference_type return_type;

		auto dist_parent = std::distance(m_itparent, that.m_itparent);

		if(dist_parent == 0)
			return std::distance(m_itchild, that.m_itchild);
		else if(dist_parent > 0)
		{
			return_type dist = std::distance(m_itchild, m_children.end());

			auto curparent = m_itparent;
			while(++curparent != that.m_itparent)
				dist += boost::size(*curparent);

			if(curparent != m_parents.end())
			{
				child_iterator itbeg(curparent->begin(), *curparent);
				dist += std::distance(itbeg, that.m_itchild);
			}

			return dist;
		}
		else 
		{
			return_type dist = std::distance(m_children.begin(), m_itchild);

			auto curparent = m_itparent;
			while(--curparent != that.m_itparent)
				dist += boost::size(*curparent);

			child_iterator itend(curparent->end(), *curparent);
			dist += std::distance(that.m_itchild, itend);

			return -dist;
		}
	}

	typename facade::reference dereference() const
	{
		assert(m_itchild != m_children.end());
		return *m_itchild;
	}

	parent_range m_parents;
	children_range m_children;

	parent_iterator m_itparent;
	child_iterator m_itchild;
};

template <class IT,int L>
typename boost::iterator_value<IT>::type flat_iterator<IT,L>::null_child;

template <class IT>
class flat_iterator<IT,0> : public IT
{
public:
	flat_iterator() : IT(0) {}

	typedef flat_iterator this_type;
	typedef flat_iterator type;

	template <class IT2, class = 
		typename std::enable_if<std::is_convertible<IT2,IT>::value>::type>
	flat_iterator(const flat_iterator<IT2,0> &that)
		: IT(that) {}

	template <class X>
	flat_iterator(IT it, X &&)
		: IT(it)
	{
	}
};

template <class C, int L=0>
class flat_view 
	: public boost::iterator_range
	<
		flat_iterator
		<
			typename boost::range_iterator<C>::type,
			L
		>
	>
{
	typedef flat_iterator
	<
		typename boost::range_iterator<C>::type,
		L
	> flat_it;

	typedef typename flat_view::this_type base;
public:
	flat_view(C &container) 
		: base(flat_it(boost::begin(container), container), 
			   flat_it(boost::end(container), container))
	{
	}
};

template <int L=0>
struct flattened {};

template <class R, int L> inline 
auto operator|(R &&r, flattened<L>)
	-> flat_view<typename std::remove_reference<R>::type,L> 
{
	return flat_view<typename std::remove_reference<R>::type,L>(r);
}

}

#endif
