// $Id: recursion_guard.h 1565 2008-09-26 02:25:21Z rodolfo $

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

#ifndef RECURSION_GUARD_H
#define RECURSION_GUARD_H
#pragma once

namespace s3d
{

class recursion_guard
{
public:
    recursion_guard();

    void lock();
    void unlock();
    bool try_lock();

private:
    recursion_guard(const recursion_guard &);
    recursion_guard &operator=(const recursion_guard &);

    struct impl;
    std::shared_ptr<impl> pimpl;
};

class scoped_recursion_guard
{
public:
    scoped_recursion_guard(recursion_guard &rg);

    bool recursion() const;
private:
    scoped_recursion_guard(const scoped_recursion_guard &);
    scoped_recursion_guard &operator=(const scoped_recursion_guard &);

    struct impl;
    std::shared_ptr<impl> pimpl;
};

} // namespace s3d

#endif

