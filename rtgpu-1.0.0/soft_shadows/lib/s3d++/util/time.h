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

#ifndef S3D_UTIL_TIME_H
#define S3D_UTIL_TIME_H

#include "optional.h"

namespace s3d
{

double get_time();
void wait(double seg);

double make_time(optional<int> year=none, 
				 optional<int> month=none, 
				 optional<int> day=none, 
				 optional<int> hour=none, 
				 optional<int> min=none, 
				 optional<int> sec=none);

} // namespace s3d

#endif

// $Id: time.h 2301 2009-06-10 00:00:59Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

