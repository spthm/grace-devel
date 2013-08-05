
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

namespace s3d { namespace color
{

template <class T, int D> template <class... ARGS, class> 
Radiance<T,D>::Radiance(T c1, ARGS... cn)/*{{{*/
	: coords_base(c1, cn...)
{
}/*}}}*/

template <class T, int D> template <class U, class>
Radiance<T,D>::Radiance(const U &c)
{
	std::fill(begin(), end(), math::map<T>(math::unmap(c)));
}

}} // namespace s3d::math

// $Id: complex.hpp 2752 2010-06-11 02:32:41Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

