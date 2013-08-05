#ifndef S3D_MATH_R2_REGION_H
#define S3D_MATH_R2_REGION_H

#include <vector>
#include <initializer_list>
#include <boost/operators.hpp>
#include "box.h"
#include "point.h"
#include "vector.h"

namespace s3d { namespace math { namespace r2
{

template <class T>
class Region
	: boost::andable<Region<T>,
	  boost::orable<Region<T>,
	  boost::xorable<Region<T>,
	  boost::equality_comparable<Region<T>,
	  boost::subtractable<Region<T>,
	  boost::additive<Region<T>, Vector<T,2>>>>>>>
{
public:
	typedef std::vector<Box<T,2>> box_list;
	typedef typename box_list::iterator iterator;
public:
	typedef typename box_list::const_iterator const_iterator;

	Region();
	Region(const Box<T,2> &b);
	Region(const std::initializer_list<Box<T,2>> &b);
	Region(const Region &that) = default;
	Region(Region &&that);

	Region &operator=(const Region &that) = default;
	Region &operator=(Region &&that);

	bool operator==(const Region &that) const;

	const_iterator begin() const { return m_boxes.begin(); }
	const_iterator end() const { return m_boxes.end(); }

	size_t size() const { return m_boxes.size(); }
	bool empty() const { return m_boxes.empty(); }
	void clear() { m_boxes.clear(); m_extents = {0,0,0,0}; }

	bool contains(const Point<T,2> &p) const;

	// 0: no overlap, 1: partial, 2: full
	int overlaps(const Box<T,2> &b) const;

	Region &operator|=(const Region &that);
	Region &operator&=(const Region &that);
	Region &operator-=(const Region &that);
	Region &operator^=(const Region &that);

	Region &operator+=(const Vector<T,2> &offset);
	Region &operator-=(const Vector<T,2> &offset)
		{ return operator+=(-offset); }

private:
	// ordem é importante
	box m_extents;
	box_list m_boxes;

	typedef void (*non_overlap_func)(box_list &boxes,
								    const_iterator it, const_iterator itend,
									T y, T h);

	typedef void (*overlap_func)(box_list &boxes,
							     const_iterator it1, const_iterator it1end,
							     const_iterator it2, const_iterator it2end, 
								 T y, T h);

	static void do_union_non_overlap(box_list &boxes, const_iterator it,
									 const_iterator itend, T y1, T y2);
	static void do_union_overlap(box_list &boxes,
								 const_iterator it1, const_iterator it1end,
								 const_iterator it2, const_iterator 
								 it2end, T y, T h);

	static void do_intersect_overlap(box_list &boxes,
									 const_iterator it1, const_iterator it1end,
									 const_iterator it2, const_iterator it2end,
									 T y, T h);

	static void do_subtract_non_overlap(box_list &boxes, const_iterator it,
									    const_iterator itend, T y1, T y2);
	static void do_subtract_overlap(box_list &boxes,
									const_iterator it1, const_iterator it1end,
									const_iterator it2, const_iterator it2end, 
									T y, T h);

	void op(const Region &that, overlap_func overlap,
					   non_overlap_func non_overlap1, 
					   non_overlap_func non_overlap2);

	iterator coalesce(box_list &boxes, 
					  iterator itprev_start, iterator itcur_start);
	void update_extents();
};

template <class T>
std::ostream &operator<<(std::ostream &out, const Region<T> &r);


typedef Region<real> region;
typedef Region<int> iregion;
typedef Region<double> dregion;
typedef Region<float> fregion;

}}} // namespace s3d::math::r2

#include "region.hpp"

#endif
