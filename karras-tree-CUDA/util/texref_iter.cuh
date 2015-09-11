/*
 * Copyright (c) 2015 Sam Thomson
 *
 *  This file is free software: you may copy, redistribute and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation, either version 2 of the License, or (at your
 *  option) any later version.
 *
 *  This file is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * This file incorporates work covered by the following copyright and
 * permission notice:
 *
 *     Copyright (c) 2011, Duane Merrill.  All rights reserved.
 *     Copyright (c) 2011-2015, NVIDIA CORPORATION.  All rights reserved.
 *
 *     Redistribution and use in source and binary forms, with or without
 *     modification, are permitted provided that the following conditions are met:
 *         * Redistributions of source code must retain the above copyright
 *           notice, this list of conditions and the following disclaimer.
 *         * Redistributions in binary form must reproduce the above copyright
 *           notice, this list of conditions and the following disclaimer in the
 *           documentation and/or other materials provided with the distribution.
 *         * Neither the name of the NVIDIA CORPORATION nor the
 *           names of its contributors may be used to endorse or promote products
 *           derived from this software without specific prior written permission.
 *
 *     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *     ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *     WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *     DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *     DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *     ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *     SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * This file is a derivative of CUB's [1] tex_ref_input_iterator.cuh [2].
 *
 * [1]: http://nvlabs.github.io/cub
 * [2]: https://github.com/NVlabs/cub/blob/113350fb19815e49c31b7ddbc6ec4d0a0299497a/cub/iterator/tex_ref_input_iterator.cuh
 * Accessed 10-09-2015
 */

#pragma once

#include <cstddef>
#include <iterator>

namespace grace {

enum {
    PRIMITIVE_TEX_UID = -1,
    NODE_TEX_UID = -2,
    LEAF_TEX_UID = -3
};

// unnamed namespace prevents external linkage of TypedTexRef::UniqueTexRef::ref
namespace {

template <typename T>
struct TypedTexRef
{
    template <int UID>
    struct UniqueTexRef
    {
        // Largest valid texture type of which T is a whole multiple.
        // Valid texture types are char, short, int, float, int2, int4 etc.
        typedef typename VectorWord<T>::type TexelType;

        static const size_t TEXELS_PER_READ = sizeof(T) / sizeof(TexelType);

        typedef texture<TexelType> TexRef;
        static TexRef ref;

        static GRACE_DEVICE T fetch(ptrdiff_t tex_offset)
        {
            TexelType buffer[TEXELS_PER_READ];

            #pragma unroll
            for (int i = 0; i < TEXELS_PER_READ; ++i)
            {
                buffer[i] = tex1Dfetch(ref, (tex_offset * TEXELS_PER_READ) + i);
            }

            return *reinterpret_cast<T*>(buffer);
        }
    };
};

// Texture reference definitions.
template <typename T>
template <int UID>
typename TypedTexRef<T>:: template UniqueTexRef<UID>::TexRef
    TypedTexRef<T>:: template UniqueTexRef<UID>::ref;


} // unnamed namespace

// This is a const iterator. Texture references are not writable!
// The iterator may only be dereferenced (*iter, iter[] and iter->) in device
// code.
// The iterator's bind() and unbind() methods may only be called in host code.
// UID should be >= 0 in general. UIDs < 0 are reserved for GRACE internals.
template <typename T, int UID = 0>
class TexRefIter
{
public:
    typedef TexRefIter<T, UID> self;
    typedef std::random_access_iterator_tag iterator_category;
    typedef ptrdiff_t difference_type;
    typedef T value_type;
    // Constness.
    typedef const T* pointer;
    typedef const T& reference;

private:
    const T* d_ptr;
    difference_type tex_offset;

    // Texture reference wrapper (old Tesla/Fermi-style textures)
    typedef typename TypedTexRef<T>:: template UniqueTexRef<UID> Tex;

public:
    GRACE_HOST cudaError_t bind(const T* ptr, const size_t bytes)
    {
        d_ptr = ptr;
        size_t offset = 0;
        cudaError_t cuerr = cudaBindTexture(&offset, Tex::ref, d_ptr,
                                            bytes);
        tex_offset = (difference_type) (offset / sizeof(T));
        return cuerr;
    }

    GRACE_HOST cudaError_t unbind()
    {
        return cudaUnbindTexture(Tex::ref);
    }

    // Read-only.
    GRACE_DEVICE reference operator*() const
    {
        return Tex::fetch(tex_offset);
    }

    // Read-only.
    GRACE_DEVICE reference operator[](difference_type i) const
    {
        return *((*this) + i);
    }

    // Read-only.
    GRACE_DEVICE pointer operator->() const
    {
        return &(*(*this));
    }

    // Prefix.
    GRACE_HOST_DEVICE self& operator++()
    {
        ++tex_offset;
        return *this;
    }

    // Postfix.
    GRACE_HOST_DEVICE self operator++(int)
    {
        self temp = *this;
        ++(*this);
        return temp;
    }

    // Prefix.
    GRACE_HOST_DEVICE self& operator--()
    {
        --tex_offset;
        return *this;
    }

    // Postfix.
    GRACE_HOST_DEVICE self operator--(int)
    {
        self temp = *this;
        --(*this);
        return temp;
    }

    GRACE_HOST_DEVICE self& operator+=(const difference_type n)
    {
        tex_offset += n;
        return *this;
    }

    GRACE_HOST_DEVICE self& operator-=(const difference_type n)
    {
        tex_offset -= n;
        return *this;
    }

    GRACE_HOST_DEVICE friend self operator+(const self& lhs,
                                            const difference_type rhs)
    {
        self temp;
        temp.d_ptr = lhs.d_ptr;
        temp.tex_offset = lhs.tex_offset + rhs;
        return temp;
    }

    GRACE_HOST_DEVICE friend self operator+(const difference_type lhs,
                                            const self rhs)
    {
        // Swap sides.
        return rhs + lhs;
    }

    GRACE_HOST_DEVICE friend self operator-(const self lhs,
                                            const difference_type rhs)
    {
        return lhs + (-rhs);
    }

    GRACE_HOST_DEVICE difference_type operator-(const self& other) const
    {
        return tex_offset - other.tex_offset;
    }

    GRACE_HOST_DEVICE friend bool operator==(const self& lhs, const self& rhs)
    {
        return (lhs.d_ptr == rhs.d_ptr) &&
               (lhs.tex_offset == rhs.tex_offset);
    }

    GRACE_HOST_DEVICE friend bool operator!=(const self& lhs, const self& rhs)
    {
        return !(lhs == rhs);
    }

    GRACE_HOST_DEVICE friend bool operator<(const self& lhs, const self& rhs)
    {
        return (rhs - lhs) > 0;
    }

    GRACE_HOST_DEVICE friend bool operator>(const self& lhs, const self& rhs)
    {
        // Swap sides.
        return rhs < lhs;
    }

    GRACE_HOST_DEVICE friend bool operator<=(const self& lhs, const self& rhs)
    {
        return !(lhs > rhs);
    }

    GRACE_HOST_DEVICE friend bool operator>=(const self& lhs, const self& rhs)
    {
        return !(lhs < rhs);
    }
};

} // namespace grace
