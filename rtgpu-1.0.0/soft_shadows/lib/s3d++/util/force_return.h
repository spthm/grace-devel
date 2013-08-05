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

#ifndef S3D_UTIL_FORCE_RETURN_H
#define S3D_UTIL_FORCE_RETURN_H

namespace s3d
{

namespace detail
{
	template <class R, class F> class force_return
	{
	public:
		force_return(const R &ret, F func)
			: m_return(ret), m_func(func) {}

		template <class...ARGS> R operator()(ARGS &&...args) const
		{
			m_func(args...);
			return m_return;
		}
	private:
		R m_return;
		F m_func;
	};
}

template <class R, class F> 
detail::force_return<R,F> force_return(const R &ret, F func)
{
	return detail::force_return<R,F>(ret, func);
}

} // namespace s3d

#endif

// $Id: force_return.h 2227 2009-05-27 02:30:46Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

