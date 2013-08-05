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

#ifndef S3D_UTIL_USER_DATA_H
#define S3D_UTIL_USER_DATA_H

#include "any.h"

namespace s3d
{

class has_user_data
{
public:
	has_user_data() {}
	has_user_data(const any &d) : m_user_data(d) {}
	has_user_data(any &&d) : m_user_data(std::move(d)) {}

	has_user_data &operator=(const has_user_data &that)
	{ 
		m_user_data = that.m_user_data; 
		return *this;
	}

	has_user_data &operator=(has_user_data &&that)
	{ 
		m_user_data = std::move(that.m_user_data);
		return *this;
	}

	void set_user_data(const any &a) { m_user_data = a; }
	void set_user_data(any &&a) { m_user_data = std::move(a); }

	const any &user_data() const { return m_user_data; }
	any &user_data() { return m_user_data; }

private:
	any m_user_data;
};

}

#endif

// $Id: user_data.h 2298 2009-06-09 15:24:25Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

