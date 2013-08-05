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

namespace s3d
{
inline lockable::lockable() /*{{{*/
    : m_lock(0) 
{
}/*}}}*/
inline bool lockable::locked() const /*{{{*/
{ 
    return m_lock != 0; 
}/*}}}*/

inline void lockable::lock() /*{{{*/
{ 
    do_lock(); 
}/*}}}*/
inline bool lockable::unlock() /*{{{*/
{ 
    return do_unlock(); 
}/*}}}*/
inline void lockable::do_lock() /*{{{*/
{ 
    ++m_lock; 
}/*}}}*/
inline bool lockable::do_unlock() /*{{{*/
{ 
    assert(locked()); 
    return --m_lock == 0; 
}/*}}}*/

inline scoped_lock::scoped_lock(lockable &lk, bool initially_locked)/*{{{*/
    : m_lockable(lk), m_lock_held(initially_locked)
{
    if(initially_locked)
	m_lockable.lock();
}/*}}}*/
inline scoped_lock::~scoped_lock()/*{{{*/
{
    if(m_lock_held)
	m_lockable.unlock();
}/*}}}*/

inline void scoped_lock::lock()/*{{{*/
{
    if(!m_lock_held)
    {
	m_lock_held = true;
	m_lockable.lock();
    }
}/*}}}*/
inline bool scoped_lock::unlock()/*{{{*/
{
    if(m_lock_held)
    {
	m_lock_held = false;
	return m_lock.unlock();
    }
    else
	return m_lock.locked();
}/*}}}*/
inline bool scoped_lock::locked() const /*{{{*/
{ 
    return m_lockable.locked(); 
}/*}}}*/

} // namespace s3d

// $Id: lockable.hpp 2227 2009-05-27 02:30:46Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

