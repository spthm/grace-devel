// $Id: visitor.cpp 1565 2008-09-26 02:25:21Z rodolfo $

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
#include "visitor.h"

namespace s3d
{

bool const_visitable::accept(visitor_base &&visitor) const
{ 
    scoped_recursion_guard rc(m_recguard);
    if(rc.recursion())
        return false;

    return do_accept(std::move(visitor));
}

bool visitable::accept(visitor_base &&visitor)
{ 
    scoped_recursion_guard rc(m_recguard);
    if(rc.recursion())
        return false;

    return do_accept(std::move(visitor));
}

bool visitable::move_accept(visitor_base &&visitor)
{ 
    scoped_recursion_guard rc(m_recguard);
    if(rc.recursion())
        return false;

    return do_move_accept(std::move(visitor));
}

} // namespace orm

