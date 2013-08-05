// $Id: recursion_guard.cpp 1565 2008-09-26 02:25:21Z rodolfo $

/*
   Copyright (c) 2008, Rodolfo Schulz de Lima 
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice, this
     list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.  
   * Neither the name of RodSoft nor the names of its contributors may be used to
     endorse or promote products derived from this software without specific
     prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
   LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
   INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
   CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

#include "pch.h"
#include <mutex>
#include <map>
#include <thread>
#include "recursion_guard.h"

namespace s3d
{

struct recursion_guard::impl
{
    typedef std::map<std::thread::id,std::shared_ptr<std::mutex> > mutex_t;
    mutex_t m_mtx;
	std::mutex m_mapmtx;
    std::mutex &mtx()/*{{{*/
    {
		std::unique_lock<std::mutex> lk(m_mapmtx);

        auto it = m_mtx.find(std::this_thread::get_id());
        if(it == m_mtx.end())
        {
            it = m_mtx.insert({std::this_thread::get_id(),
							   std::make_shared<std::mutex>()}).first;
        }
        return *it->second;
    }/*}}}*/
};

recursion_guard::recursion_guard()/*{{{*/
    : pimpl(new impl())
{
}/*}}}*/
void recursion_guard::lock()/*{{{*/
{
    pimpl->mtx().lock();
}/*}}}*/
void recursion_guard::unlock()/*{{{*/
{
    pimpl->mtx().unlock();
}/*}}}*/
bool recursion_guard::try_lock()/*{{{*/
{
    return pimpl->mtx().try_lock();
}/*}}}*/

struct scoped_recursion_guard::impl/*{{{*/
{
    impl(recursion_guard &rg) : lock(rg, std::defer_lock) {}
    std::unique_lock<recursion_guard> lock;
};/*}}}*/
scoped_recursion_guard::scoped_recursion_guard(recursion_guard &rg)/*{{{*/
    : pimpl(new impl(rg))
{
    pimpl->lock.try_lock();
}/*}}}*/
bool scoped_recursion_guard::recursion() const /*{{{*/
{ 
    return !pimpl->lock; 
}/*}}}*/

} // namespace s3d

