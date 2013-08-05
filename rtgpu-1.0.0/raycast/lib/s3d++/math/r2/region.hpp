#include <boost/foreach.hpp>
#include <algorithm>

namespace s3d { namespace math { namespace r2
{

template <class T>
Region<T>::Region()/*{{{*/
	: m_extents{0,0,0,0}
{
}/*}}}*/

template <class T>
Region<T>::Region(const Box<T,2> &b)/*{{{*/
	: m_extents{b}
	, m_boxes{b}
{
}/*}}}*/

template <class T>
Region<T>::Region(const std::initializer_list<Box<T,2>> &list)/*{{{*/
{
	// TODO: this is not efficient...
	for(auto it=list.begin(); it!=list.end(); ++it)
		*this |= *it;
}/*}}}*/

template <class T>
Region<T>::Region(Region &&that)/*{{{*/
	: m_extents(std::move(that.m_extents))
	, m_boxes(std::move(that.m_boxes))
{
}/*}}}*/

template <class T>
Region<T> &Region<T>::operator=(Region &&that)/*{{{*/
{
	m_extents = std::move(that.m_extents);
	m_boxes = std::move(that.m_boxes);
	return *this;
}/*}}}*/

template <class T>
void Region<T>::update_extents()/*{{{*/
{
	if(empty())
	{
		m_extents = null_box<T,2>();
		return;
	}

	auto l = lower(m_boxes.front()),
		 u = upper(m_boxes.back());

	BOOST_FOREACH(auto &r, m_boxes)
	{
		l.x = min(l.x, r.x);
		u.x = max(u.x, r.x+r.w);
	}

	assert(l.x < u.x);
	assert(l.y < u.y);

	m_extents = Box<T,2>(l,u);
}/*}}}*/

template <class T>
auto Region<T>::coalesce(box_list &boxes, /*{{{*/
				   iterator itprev_start, 
				   iterator itcur_start) -> iterator
{
	auto itprev = itprev_start, // current box in previous band
		 it = itcur_start;		// current box in current band
	auto itend = boxes.end();	// end of region

	// number of boxes in prev band
	size_t prev_count = distance(itprev_start, itcur_start);
	T band_y = it->y;					// y1 coord for current band

	size_t cur_count;
	for(cur_count=0; it != itend && it->y == band_y; ++cur_count)
		++it;

	/*  
	 * Figure out how many rectangles are in the current band. Have to do this
	 * because multiple bands could have been added in miRegionOp at the end
	 * when one region has been exhausted
	 */
	if(it != itend)
	{
		/*
		 * If more than one band was added, we have to find the start
		 * of the last band added so the next coalescing job can start
		 * at the right place... (given when multiple bands are added,
		 * this may be pointless -- see above).
		 */

		itcur_start = itend;
		do
		{
			--itcur_start;
		}
		while(std::prev(itcur_start)->y == itcur_start->y);

		itend = boxes.end();
	}

	if(cur_count == prev_count && cur_count != 0)
	{
		it -= cur_count;

		 /*
		  * The bands may only be coalesced if the bottom of the previous
		  * matches the top scanline of the current.
		  */
		if(itprev->y+itprev->h == it->y)
		{
			/*
			 * Make sure the bands have boxes in the same places. This
			 * assumes that boxes have been added in such a way that they
			 * cover the most area possible. I.e. two boxes in a band must
			 * have some horizontal space between them.
			 */
			do
			{
				if(itprev->x != it->x || itprev->x+itprev->w != it->x+it->w)
				{
                    // The bands don't line up so they can't be coalesced.
					return itcur_start;
				}

				itprev++;
				it++;
			}
			while(--prev_count != 0);

			it -= cur_count;
			itprev -= cur_count;
			boxes.resize(boxes.size() - cur_count);

            /*
             * The bands may be merged, so set the bottom y of each box
             * in the previous band to that of the corresponding box in
             * the current band.
             */
			do
			{
				itprev->h = it->y+it->h - itprev->y;
				itprev++;
				it++;
			}
			while(--cur_count != 0);

            /*
             * If only one band was added to the region, we have to backup
             * curStart to the start of the previous band.
             *
             * If more than one band was added to the region, copy the
             * other bands down. The assumption here is that the other bands
             * came from the same region as the current one and no further
             * coalescing can be done on them since it's all been done
             * already... curStart is already in the right place.
             */
			if(it == itend)
				itcur_start = itprev_start;
			else
				copy(it, itend, itprev);
		}
	}

	return itcur_start;
}/*}}}*/

template <class T>
void Region<T>::op(const Region &that, /*{{{*/
				   overlap_func overlap,
				   non_overlap_func non_overlap1, 
				   non_overlap_func non_overlap2)
{
	auto it1 = m_boxes.begin(),			// first region range
		 it1end = m_boxes.end(), 

		 it2 = that.m_boxes.begin(),	// second region range
		 it2end = that.m_boxes.end();

	box_list newreg;
	newreg.reserve(max(m_boxes.size(), that.m_boxes.size())*2);

    /*
     * Initialize ybot and ytop.
     * In the upcoming loop, ybot and ytop serve different functions depending
     * on whether the band being handled is an overlapping or non-overlapping
     * band.
     * In the case of a non-overlapping band (only one of the regions
     * has points in the band), ybot is the bottom of the most recent
     * intersection and thus clips the top of the rectangles in that band.
     * ytop is the top of the next intersection between the two regions and
     * serves to clip the bottom of the rectangles in the current band.
     * For an overlapping band (where the two regions intersect), ytop clips
     * the top of the rectangles of both regions and ybot clips the bottoms.
     */
	T ybot = min(m_extents.y, that.m_extents.y); // bottom of intersection

    /*
     * prevBand serves to mark the start of the previous band so rectangles
     * can be coalesced into larger rectangles. qv. miCoalesce, above.
     * In the beginning, there is no previous band, so prevBand == curBand
     * (curBand is set later on, of course, but the first band will always
     * start at index 0). prevBand and curBand must be indices because of
     * the possible expansion, and resultant moving, of the new region's
     * array of rectangles.
     */
	auto itprev_band = newreg.begin();	// start of previous band in newreg

	do
	{
        /*
         * This algorithm proceeds one source-band (as opposed to a
         * destination band, which is determined by where the two regions
         * intersect) at a time. r1BandEnd and r2BandEnd serve to mark the
         * rectangle after the last one in the current band for their
         * respective regions.
         */

		auto it1_band_end = it1; // end of band in r1
		while(it1_band_end != it1end && it1_band_end->y == it1->y)
			++it1_band_end;

		auto it2_band_end = it2; // end of band in r2
		while(it2_band_end != it2end && it2_band_end->y == it2->y)
			++it2_band_end;

		T ytop, // top of intersection
		  top,  // top of non-overlapping band
		  bot;  // bottom of non-overlapping band

		size_t oldsize = newreg.size();

        /*
         * First handle the band that doesn't intersect, if any.
         *
         * Note that attention is restricted to one band in the
         * non-intersecting region at once, so if a region has n
         * bands between the current position and the next place it overlaps
         * the other, this entire loop will be passed through n times.
         */
		if(it1->y < it2->y)
		{
			top = max(it1->y, ybot);
			bot = min(it1->y+it1->h, it2->y);

			if(top != bot && non_overlap1)
				non_overlap1(newreg, it1, it1_band_end, top, bot-top);

			ytop = it2->y;
		}
		else if(it2->y < it1->y)
		{
			top = max(it2->y, ybot);
			bot = min(it2->y+it2->h, it1->y);

			if(top != bot && non_overlap2)
				non_overlap2(newreg, it2, it2_band_end, top, bot-top);

			ytop = it1->y;
		}
		else
			ytop = it1->y;

		// start of current band in newreg
		auto itcur_band = newreg.begin()+oldsize;

        /*
         * If any rectangles got added to the region, try and coalesce them
         * with rectangles from the previous band. Note we could just do
         * this test in miCoalesce, but some machines incur a not
         * inconsiderable cost for function calls, so...
         */
		if(newreg.end() != itcur_band)
			itprev_band = coalesce(newreg, itprev_band, itcur_band);

        /*
         * Now see if we've hit an intersecting band. The two bands only
         * intersect if ybot > ytop
         */
		ybot = min(it1->y+it1->h, it2->y+it2->h);
		oldsize = newreg.size();

		if(ybot > ytop)
			overlap(newreg, it1, it1_band_end, it2, it2_band_end, ytop, ybot-ytop);

		itcur_band = newreg.begin()+oldsize;

		if(itcur_band != newreg.end())
			itprev_band = coalesce(newreg, itprev_band, itcur_band);

        /*
         * If we've finished with a band (y2 == ybot) we skip forward
         * in the region to the next band.
         */
		if(it1->y+it1->h == ybot)
			it1 = it1_band_end;

		if(it2->y+it2->h == ybot)
			it2 = it2_band_end;
	}
	while(it1 != it1end && it2 != it2end);

    // Deal with whichever region still has rectangles left.
	size_t oldsize = newreg.size();

	if(it1 != it1end)
	{
		if(non_overlap1)
		{
			do
			{
				auto it1_band_end = it1;
				while(it1_band_end != it1end && it1_band_end->y == it1->y)
					++it1_band_end;

				T y = max(it1->y, ybot);

				non_overlap1(newreg, it1, it1_band_end, 
							 y, it1->y+it1->h - y);

				it1 = it1_band_end;
			}
			while(it1 != it1end);
		}
	}
	else if(it2 != it2end && non_overlap2)
	{
		do
		{
			auto it2_band_end = it2;
			while(it2_band_end != it2end && it2_band_end->y == it2->y)
				++it2_band_end;

			T y = max(it2->y, ybot);

			non_overlap2(newreg, it2, it2_band_end, 
						 y, it2->y+it2->h - y);

			it2 = it2_band_end;
		}
		while(it2 != it2end);
	}

	auto itcur_band = newreg.begin()+oldsize;

	if(newreg.end() != itcur_band)
		coalesce(newreg, itprev_band, itcur_band);

	m_boxes = std::move(newreg);
}/*}}}*/

template <class T>
void Region<T>::do_union_non_overlap(box_list &boxes, const_iterator it, /*{{{*/
									 const_iterator itend, T y, T h)
{
	assert(h > 0);

	while(it != itend)
	{
		assert(it->w > 0);
		boxes.push_back(Box<T,2>{it->x, y, it->w, h});

		++it;
	}
}/*}}}*/

template <class T>
void Region<T>::do_union_overlap(box_list &boxes, /*{{{*/
								 const_iterator it1, const_iterator it1end,
								 const_iterator it2, const_iterator it2end, 
								 T y, T h)
{
	struct aux
	{
		static void merge_box(const_iterator it, box_list &boxes, T y, T h)
		{
			auto itprev = prev(boxes.end());

			if(!boxes.empty() && 
			   itprev->y == y &&
			   itprev->h == h &&
			   itprev->x+itprev->w >= it->x)
			{
				if(itprev->x+itprev->w < it->x+it->w)
				{
					itprev->w = it->x+it->w - itprev->x;
					assert(itprev->w > 0);
				}
			}
			else
				boxes.push_back(Box<T,2>{it->x, y, it->w, h});
		}
	};

	assert(h > 0);
	while(it1 != it1end && it2 != it2end)
	{
		if(it1->x < it2->x)
			aux::merge_box(it1++, boxes, y, h);
		else
			aux::merge_box(it2++, boxes, y, h);
	}

	if(it1 != it1end)
	{
		do
		{
			aux::merge_box(it1++, boxes, y, h);
		}
		while(it1 != it1end);
	}
	else
	{
		while(it2 != it2end)
			aux::merge_box(it2++, boxes, y, h);
	}
}/*}}}*/

template <class T>
void Region<T>::do_intersect_overlap(box_list &boxes, /*{{{*/
								     const_iterator it1, const_iterator it1end,
								     const_iterator it2, const_iterator it2end,
									 T y, T h)
{
	while(it1 != it1end && it2 != it2end)
	{
		T x1 = max(it1->x, it2->x),
		  x2 = min(it1->x+it1->w, it2->x+it2->w);

		if(x1 < x2)
		{
			assert(h > 0);
			boxes.push_back(Box<T,2>{x1,y,x2-x1,h});
		}

		if(it1->x+it1->w < it2->x+it2->w)
			++it1;
		else if(it2->x+it2->w < it1->x+it1->w)
			++it2;
		else
		{
			++it1;
			++it2;
		}
	}
}/*}}}*/

template <class T>
void Region<T>::do_subtract_non_overlap(box_list &boxes, /*{{{*/
										const_iterator it,const_iterator itend, 
										T y, T h)
{
	assert(h > 0);

	while(it != itend)
	{
		assert(it->w > 0);
		boxes.push_back(Box<T,2>{it->x, y, it->w, h});

		++it;
	}
}/*}}}*/

template <class T>
void Region<T>::do_subtract_overlap(box_list &boxes, /*{{{*/
								    const_iterator it1, const_iterator it1end,
								    const_iterator it2, const_iterator it2end, 
									T y, T h)
{
	T x1 = it1->x;

	assert(h > 0);

	while(it1 != it1end && it2 != it2end)
	{
		if(it2->x+it2->w <= x1)
		{
			// subtrahend missed the boat: go to next subtrahend
			it2++;
		}
		else if(it2->x <= x1)
		{
			// subtrahend preceeds minuend: nuke left edge of minuend

			x1 = it2->x+it2->w;
			if(x1 >= it1->x+it1->w)
			{
				// minuend completely covered: advance to next minuend and
				// reset left fence to edge of new minuend
				it1++;
				if(it1 != it1end)
					x1 = it1->x;
			}
			else
			{
				// subtrahend now used up since it doesn't extend beyond minuend
				it2++;
			}
		}
		else if(it2->x < it1->x+it1->w)
		{
			// Left part of subtrahend covers part of minuend: add uncovered
			// part of minuend to region and skip to next subtrahend
			assert(x1 < it2->x);
			boxes.push_back(Box<T,2>{x1,y,it2->x-x1,h});

			x1 = it2->x+it2->w;
			if(x1 >= it1->x+it1->w)
			{
				it1++; // minuend used up: advance to new...

				if(it1 != it1end)
					x1 = it1->x;
			}
			else
				it2++; // subtrahend used up
		}
		else
		{
			// minuend used up: add any remaining piece before advancing
			if(it1->x+it1->w > x1)
				boxes.push_back(Box<T,2>{x1,y,it1->x+it1->w-x1,h});

			++it1;
			if(it1 != it1end)
				x1 = it1->x;
		}
	}

	// add remaining minuend rectangles to region
	while(it1 != it1end)
	{
		assert(x1 < it1->x+it1->w);
		boxes.push_back(Box<T,2>{x1,y,it1->x+it1->w-x1,h});

		++it1;
		if(it1 != it1end)
			x1 = it1->x;
	}
}/*}}}*/

template <class T>
Region<T> &Region<T>::operator |=(const Region<T> &that)/*{{{*/
{
	if(this == &that || that.empty())
		return *this;

	if(empty())
	{
		*this = that;
		return *this;
	}

	if(m_boxes.size()==1 && m_extents.contains(that.m_extents))
		return *this;

	if(that.m_boxes.size()==1 && that.m_extents.contains(m_extents))
	{
		*this = that;
		return *this;
	}

	op(that, do_union_overlap, do_union_non_overlap, do_union_non_overlap);

	m_extents |= that.m_extents;
	return *this;
};/*}}}*/

template <class T>
Region<T> &Region<T>::operator &=(const Region<T> &that)/*{{{*/
{
	if(this == &that || empty())
		return *this;

	if(empty())
		return *this;

	if(that.empty() || !overlap(m_extents,that.m_extents))
	{
		clear();
		return *this;
	}

	if(m_boxes.size()==1 && m_extents.contains(that.m_extents))
	{
		*this = that;
		return *this;
	}

	if(that.m_boxes.size()==1 && that.m_extents.contains(m_extents))
		return *this;

	op(that, do_intersect_overlap, NULL, NULL);

	update_extents();

	return *this;
};/*}}}*/

template <class T>
Region<T> &Region<T>::operator -=(const Region<T> &that)/*{{{*/
{
	if(that.empty() || empty() || !overlap(m_extents,that.m_extents))
		return *this;

	op(that, do_subtract_overlap, do_subtract_non_overlap, NULL);

	update_extents();
	return *this;
}/*}}}*/

template <class T>
Region<T> &Region<T>::operator ^=(const Region<T> &that)/*{{{*/
{
	auto aux = that;
	aux -= *this;
	*this -= that;
	*this |= aux;
	return *this;
}/*}}}*/

template <class T>
Region<T> &Region<T>::operator +=(const Vector<T,2> &offset)/*{{{*/
{
	BOOST_FOREACH(auto &r, m_boxes)
		r += offset;

	m_extents += offset;
	return *this;
}/*}}}*/

template <class T>
bool Region<T>::operator==(const Region &that) const/*{{{*/
{
	if(m_boxes.size() != that.m_boxes.size())
		return false;
	else if(empty())
		return true;
	else if(m_extents != that.m_extents)
		return false;
	else
	{
		for(size_t i=0; i<m_boxes.size(); ++i)
		{
			if(m_boxes[i] != that.m_boxes[i])
				return false;
		}
	}
	return true;
}/*}}}*/

template <class T>
bool Region<T>::contains(const Point<T,2> &p) const/*{{{*/
{
	if(empty())
		return false;
	else if(!m_extents.contains(p))
		return false;
	else
	{
		BOOST_FOREACH(auto &b, m_boxes)
		{
			if(b.contains(p))
				return true;
		}
	}
	return false;
}/*}}}*/

template <class T>
int Region<T>::overlaps(const Box<T,2> &test) const/*{{{*/
{
	if(empty())
		return 0;

	T rx = test.x,
	  ry = test.y;

	if(!overlap(m_extents,test))
		return 0;

	bool part_in = false, part_out = false;

	// can stop when both part_out and part_in are true, or we reach r.y2
	BOOST_FOREACH(auto &b, m_boxes)
	{
		if(b.y+b.h <= ry)
			continue; // getting up to speed or skipping remainder of band

		if(b.y > ry)
		{
			part_out = true; // missed part of rectangle above
			if(part_in || (b.y >= test.y+test.h))
				break;
			ry = b.y; // x guaranteed to be == test.x
		}

		if(b.x+b.w <= rx)
			continue; // not far enough over yet

		if(b.x > rx)
		{
			part_out = true; // missed part of rectangle to left
			if(part_in)
				break;
		}

		if(b.x < test.x+test.w)
		{
			part_in = true;
			if(part_out)
				break;
		}

		if(b.x+b.w >= test.x+test.w)
		{
			ry = b.y+b.h; // finished with this band
			if(ry >= test.y+test.h)
				break;
			rx = test.x; // reset x out to left again
		}
		else
		{
			/*
			 * Because boxes in a band are maxima width, if the first box
			 * to overlap the rectangle doesn't completely cover it in that
			 * band, the rectangle must be partially out, since some of it
			 * will be uncovered in that band. part_in will have been set true
			 * by now...
			 */
		}
	}

	if(part_in)
	{
		if(ry < test.y+test.h)
			return 1; // partial
		else
			return 2; // total
	}
	else
		return false;
}/*}}}*/

template <class T>
std::ostream &operator<<(std::ostream &out, const Region<T> &r)/*{{{*/
{
	out << "{";

	size_t c=0;
	BOOST_FOREACH(auto &box, r)
	{
		out << box;
		if(++c < r.size())
			out << ";";
	}

	return out << "}";
}/*}}}*/

}}} // namespace s3d::math::r2
