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

#ifndef S3D_SCOPED_VALUE
#define S3D_SCOPED_VALUE

#include <memory>

namespace s3d
{

class scoped_value
{
	// Type hiding technique to free the user from passing the variable type
	// explicitly
	class value_base/*{{{*/
	{
	public:
		virtual void revert() = 0;
	};/*}}}*/
	template <class T> class value : public value_base/*{{{*/
	{
	public:
		value(T &var, T value)
			: m_var(var)
			  , m_oldval(var)
		{
			m_var = value;
		}
		value(T &var) : m_var(var) , m_oldval(var) {}

		virtual void revert() { m_var = m_oldval; }

	private:
		T &m_var;
		T m_oldval;
	};/*}}}*/

public:
	scoped_value() {}
	scoped_value(scoped_value &&that) : m_value(std::move(that.m_value)) {}
	template <class T, class U> scoped_value(T &var, U val)
		: m_value(new value<T>(var, val)) {}

	template <class T> scoped_value(T &var)
		: m_value(new value<T>(var)) {}

	~scoped_value()
	{
		m_value->revert();
	}

	scoped_value &operator=(scoped_value &&that)
	{
		m_value = std::move(that.m_value);
		return *this;
	}

private:
	scoped_value(const scoped_value &);
	void operator=(const scoped_value &);
	std::shared_ptr<value_base> m_value;
};

} // namespace s3d

#endif

// $Id: scoped_value.h 2386 2009-06-29 22:35:29Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

