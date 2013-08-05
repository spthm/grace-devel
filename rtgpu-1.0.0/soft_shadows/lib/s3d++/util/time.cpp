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

#include "pch.h"
#include "time.h"

namespace s3d
{

double get_time()
{
    struct timeval tp;
    gettimeofday(&tp, 0);
    return tp.tv_sec+tp.tv_usec/1e6;
}

double make_time(optional<int> year, optional<int> month, optional<int> day, 
				 optional<int> hour, optional<int> min, optional<int> sec)
{
	time_t now;
	time(&now);
	struct tm *t = localtime(&now);

	if(year)
		t->tm_year = *year;
	if(month)
		t->tm_mon = *month;
	if(day)
		t->tm_mday = *day;
	if(hour)
		t->tm_hour = *hour;
	if(min)
		t->tm_min = *min;
	if(sec)
		t->tm_sec = *sec;

	return mktime(t);
}

void wait(double seg)
{
    usleep(static_cast<unsigned>(seg*1e6));
}

} // namespace s3d

// $Id: time.cpp 2301 2009-06-10 00:00:59Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

