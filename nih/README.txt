
--- Description ------------------------------------------------------------------------------

 nih is a simple library for performing _all kinds of basic graphics stuff_ efficiently
 that the author wouldn't want *not* to be *invented here* - and that will necessarily be
 such for anybody else but him. ;)
 Truth is, this tiny little name just sounded good as a short, easy namespace.

 Besides simple classes for manipulating small objects from linear algebra and monte carlo
 and quasi-monte carlo sampling, it contains fast algorithms for point cloud manipulation,
 hierarchy building and stuff like that.

 As a library, it's supposed to be highly modular: one can simply include a subset of its
 modules in its own project and everything should work just as well (as long as dependencies
 are respected).

--- External Dependencies --------------------------------------------------------------------

 - CUDA 4.1

 - Thrust

 - back40computing   -   http://code.google.com/p/back40computing/

--- License ----------------------------------------------------------------------------------

 Copyright (c) 2010-2011, NVIDIA Corporation
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:
   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
   * Neither the name of NVIDIA Corporation nor the
     names of its contributors may be used to endorse or promote products
     derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
