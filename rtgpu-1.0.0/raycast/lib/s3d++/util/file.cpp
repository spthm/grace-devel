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
#include "file.h"

namespace s3d
{

std::string read_stream(std::istream &in)
{
	std::stringstream ss;

	std::string line;
	while(getline(in, line))
		ss << line << '\n';

	return ss.str();
}

std::string read_file(const std::string &fname)
{
	std::ifstream in(fname.c_str());
	if(!in)
		throw std::runtime_error(fname + ": " + strerror(errno));

	return read_stream(in);
}

}

// $Id: file.cpp 2419 2009-07-09 12:37:48Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

