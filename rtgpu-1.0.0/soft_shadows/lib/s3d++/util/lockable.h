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

#ifndef S3D_UTIL_LOCKABLE_H
#define S3D_UTIL_LOCKABLE_H

namespace s3d
{

class lockable
{
public:
    lockable();
    bool locked();

    void lock();
    bool unlock();
protected:
    virtual void do_lock();
    virtual bool do_unlock();
private:
    int m_lock;
};

class scoped_lock
{
public:
    scoped_lock(lockable &lk, bool initially_locked=true);
    ~scoped_lock();

    void lock();
    bool unlock();
    bool locked() const;
private:
    bool m_lock_held;
    lockable &m_lockable;
};

} // namespace s3d

#include "lockable.hpp"

#endif

// $Id: lockable.h 2227 2009-05-27 02:30:46Z rodolfo $
// vim: nocp noet ci sts=4 fdm=marker fmr={{{,}}}
// vi: ai sw=4 ts=4

