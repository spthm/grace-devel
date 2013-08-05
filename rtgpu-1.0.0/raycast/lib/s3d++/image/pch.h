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

#include "config.h"
#include <sys/stat.h>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <cfloat> // pro FLT_MAX
#include <stdexcept>
#include <vector>
#include <fstream>
#include <cerrno>
#include <numeric>
#include <functional>
#include <mutex>

#include <boost/optional.hpp>
#include <boost/scoped_array.hpp>
#include <boost/foreach.hpp>
#include <boost/cstdint.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/iostreams/stream.hpp>

#if HAS_JPEG
extern "C" 
{
#	include <jpeglib.h>
}
#endif

#if HAS_PNG
#	include <png.h>
#endif


#include "../math/external_pch.h"

// $Id: pch.h 2946 2010-08-10 00:01:19Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

