/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*! \file morton.h
 *   \brief Defines some general purpose algorithms.
 */

#pragma once

#include <nih/basic/types.h>

namespace nih {

/// compute the Morton code of a given 3d point
///
/// \param x    x component
/// \param y    y component
/// \param z    z component
FORCE_INLINE NIH_HOST_DEVICE uint32 morton_code(
    uint32 x,
    uint32 y,
    uint32 z)
{
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x <<  8)) & 0x0300F00F;
    x = (x | (x <<  4)) & 0x030C30C3;
    x = (x | (x <<  2)) & 0x09249249;

    y = (y | (y << 16)) & 0x030000FF;
    y = (y | (y <<  8)) & 0x0300F00F;
    y = (y | (y <<  4)) & 0x030C30C3;
    y = (y | (y <<  2)) & 0x09249249;

    z = (z | (z << 16)) & 0x030000FF;
    z = (z | (z <<  8)) & 0x0300F00F;
    z = (z | (z <<  4)) & 0x030C30C3;
    z = (z | (z <<  2)) & 0x09249249;

    return x | (y << 1) | (z << 2);
}

/// compute the 60-bit Morton code of a given 3d point
///
/// \param x    x component
/// \param y    y component
/// \param z    z component
FORCE_INLINE NIH_HOST_DEVICE uint64 morton_code60(
    uint32 x,
    uint32 y,
    uint32 z)
{
    uint32 lo_x = x & 1023u;
    uint32 lo_y = y & 1023u;
    uint32 lo_z = z & 1023u;
    uint32 hi_x = x >> 10u;
    uint32 hi_y = y >> 10u;
    uint32 hi_z = z >> 10u;

    return
        (uint64(morton_code( hi_x, hi_y, hi_z )) << 30) |
         uint64(morton_code( lo_x, lo_y, lo_z ));
}


/// a convenience functor to compute the Morton code of a point sequences
/// relative to a given bounding box
template <typename Integer>
struct morton_functor {};

/// a convenience functor to compute the Morton code of a point sequences
/// relative to a given bounding box
template <>
struct morton_functor<uint32>
{
    /// constructor
    ///
    /// \param bbox     global bounding box
    NIH_HOST_DEVICE morton_functor(const Bbox3f& bbox) :
        m_base( bbox[0] ),
        m_inv(
            1.0f / (bbox[1][0] - bbox[0][0]),
            1.0f / (bbox[1][1] - bbox[0][1]),
            1.0f / (bbox[1][2] - bbox[0][2]) )
    {}

    template <typename Point_type>
    FORCE_INLINE NIH_HOST_DEVICE uint32 operator() (const Point_type point) const
    {
        //! quantize maps a float in [0,1] to an integer in [0,n).
        //! For 32-bit input, we construct a 30-bit key from three 10-bit
        //! inputs, hence map to integers in the range [0,2^10=1024)
        uint32 x = quantize( (point[0] - m_base[0]) * m_inv[0], 1024u );
        uint32 y = quantize( (point[1] - m_base[1]) * m_inv[1], 1024u );
        uint32 z = quantize( (point[2] - m_base[2]) * m_inv[2], 1024u );

        return morton_code( x, y, z );
    }

    const Vector3f m_base;
    const Vector3f m_inv;
};

/// a convenience functor to compute the Morton code of a point sequences
/// relative to a given bounding box
template <>
struct morton_functor<uint64>
{
    /// constructor
    ///
    /// \param bbox     global bounding box
    NIH_HOST_DEVICE morton_functor(const Bbox3f& bbox) :
        m_base( bbox[0] ),
        m_inv(
            1.0f / (bbox[1][0] - bbox[0][0]),
            1.0f / (bbox[1][1] - bbox[0][1]),
            1.0f / (bbox[1][2] - bbox[0][2]) )
    {}

    template <typename Point_type>
    FORCE_INLINE NIH_HOST_DEVICE uint64 operator() (const Point_type point) const
    {
        uint32 x = quantize( (point[0] - m_base[0]) * m_inv[0], 1u << 20 );
        uint32 y = quantize( (point[1] - m_base[1]) * m_inv[1], 1u << 20 );
        uint32 z = quantize( (point[2] - m_base[2]) * m_inv[2], 1u << 20 );

        return morton_code60( x, y, z );
    }

    const Vector3f m_base;
    const Vector3f m_inv;
};

} // namespace nih
