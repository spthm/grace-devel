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

#ifndef S3D_COLOR_NAMES_H
#define S3D_COLOR_NAMES_H

#include "rgb.h"

namespace s3d { namespace color
{

const rgb WHITE(1,1,1),
          BLACK(0,0,0),
		  GRAY(0.5,0.5,0.5),

		  RED(1,0,0),
		  GREEN(0,1,0),
		  BLUE(0,0,1),
		  YELLOW(1,1,0),
		  CYAN(0,1,1),
		  MAGENTA(1,0,1),

		  ORANGE(1,0.5,0),
		  BROWN(0.6, 0.3, 0);
}} // namespace s3d

#endif

// $Id: names.h 2405 2009-07-08 03:04:49Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

