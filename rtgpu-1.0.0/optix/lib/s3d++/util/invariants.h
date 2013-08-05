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

#ifndef S3D_UTIL_INVARIANTS_H
#define S3D_UTIL_INVARIANTS_H

namespace s3d
{

class has_invariants
{
public:
#ifdef NDEBUG
	void check_invariants() const {}
#else
	void check_invariants() const { do_check_invariants(); }
#endif

private:
	virtual void do_check_invariants() const = 0;
};

class scoped_invariants_check/*{{{*/
{
public:
	scoped_invariants_check(const has_invariants &obj) : m_obj(obj)
	{
		m_obj.check_invariants();
	}
	~scoped_invariants_check()
	{
		m_obj.check_invariants();
	}

private:
	const has_invariants &m_obj;
};/*}}}*/
class scoped_dtor_invariants_check/*{{{*/
{
public:
	scoped_dtor_invariants_check(const has_invariants &obj) : m_obj(obj)
	{
		m_obj.check_invariants();
	}

private:
	const has_invariants &m_obj;
};/*}}}*/
class scoped_ctor_invariants_check/*{{{*/
{
public:
	scoped_ctor_invariants_check(const has_invariants &obj) : m_obj(obj)
	{
	}
	~scoped_ctor_invariants_check()
	{
		m_obj.check_invariants();
	}
private:
	const has_invariants &m_obj;
};/*}}}*/

} // namespace s3d

#endif

// $Id: invariants.h 2227 2009-05-27 02:30:46Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

