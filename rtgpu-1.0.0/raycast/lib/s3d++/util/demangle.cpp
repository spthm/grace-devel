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

#include "pch.h"
#include "demangle.h"

namespace s3d
{

std::string demangle(const std::string &s)
{
	int status;

	char *demangled = abi::__cxa_demangle(s.c_str(), NULL, NULL, &status);

	if(demangled)
	{
		std::string ret = demangled;
		free(demangled);
		return ret;
	}
	else
		return s;
}

} 

// $Id: demangle.cpp 2266 2009-06-04 02:46:14Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

