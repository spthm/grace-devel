/*
	Copyright (c) 2009, Rodolfo Schulz de Lima.

	This file is part of S3D++.

	S3D++ is free software: you can redistribute it and/or modify 
	it under the terms of the GNU Lesser General Public License version 3 as 
	published by the Free Software Foundation.

	S3D++ is distributed in the hope that it will be useful, 
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the 
	GNU Lesser General Public License for more details.

	You should have received a copy of the GNU Lesser General Public License
	along with S3D++.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef S3D_UTIL_FORMAT_H
#define S3D_UTIL_FORMAT_H

#include <boost/format.hpp>

namespace s3d
{

namespace detail
{
	template <class T> boost::format &build_format(boost::format &fmt, 
												   const T &a)
	{
		return fmt % a;
	}
	template <class T> boost::format &build_format(boost::format &fmt)
	{
		return fmt;
	}

	template <class T, class... ARGS> 
	boost::format &build_format(boost::format &fmt, const T &a,
								const ARGS &...args)
	{
		return build_format(fmt % a, args...);
	}
}

template <class... ARGS> std::string format(const std::string &fmt,const ARGS &...args)
{
	boost::format format(fmt);

	return str(detail::build_format(format, args...));
}

} // namespace s3d

#endif

// $Id: format.h 2227 2009-05-27 02:30:46Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

