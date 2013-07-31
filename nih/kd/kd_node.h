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

/*! \file kd_node.h
 *   \brief Define basic k-d tree node structure.
 */

#pragma once

#include <nih/basic/types.h>

namespace nih {

/*! \addtogroup kdtree k-d Trees
 *  \{
 */

///
/// A k-d tree node.
/// A node can either be a leaf and have no children, or be
/// an internal split node. If a split node, its children
/// will be consecutive in memory.
/// Supports up to 7 dimensions.
///
struct Kd_node
{
    static const uint32 kInvalid = uint32(-1);

    /// empty constructor
    ///
    NIH_HOST_DEVICE Kd_node() {}

    /// leaf constructor
    ///
    /// \param index    child index
    NIH_HOST_DEVICE Kd_node(uint32 index) :
        m_packed_info( 7u | (index << 3) ) {}

    /// split node constructor
    ///
    /// \param split_dim    splitting dimension
    /// \param split_plane  splitting plane
    /// \param index        child index
    NIH_HOST_DEVICE Kd_node(const uint32 split_dim, const float split_plane, uint32 index) :
        m_packed_info( split_dim | (index << 3) ),
        m_split_plane( split_plane ) {}

    /// is a leaf?
    ///
    NIH_HOST_DEVICE uint32 is_leaf() const
    {
        return (m_packed_info & 7u) == 7u;
    }
    /// get offset of the first child
    ///
    NIH_HOST_DEVICE uint32 get_child_offset() const
    {
        return m_packed_info >> 3u;
    }
    /// get leaf index
    ///
    NIH_HOST_DEVICE uint32 get_leaf_index() const
    {
        return m_packed_info >> 3u;
    }
    /// get i-th child
    ///
    /// \param i    child index
    NIH_HOST_DEVICE uint32 get_child(const uint32 i) const
    {
        return get_child_offset() + i;
    }
    /// is the i-th child active?
    ///
    /// \param i    child index
    NIH_HOST_DEVICE bool has_child(const uint32 i) const
    {
        return is_leaf() ? false : true;
    }
    /// get left partition (or kInvalid if not active)
    ///
    NIH_HOST_DEVICE uint32 get_left() const
    {
        return get_child_offset();
    }
    /// get right partition (or kInvalid if not active)
    ///
    NIH_HOST_DEVICE uint32 get_right() const
    {
        return get_child_offset() + 1u;
    }

    /// get splitting dimension
    ///
    NIH_HOST_DEVICE uint32 get_split_dim() const { return (m_packed_info & 7u); }

    /// get splitting plane
    ///
    NIH_HOST_DEVICE float get_split_plane() const { return m_split_plane; }

    uint32 m_packed_info;
    float  m_split_plane;
};

/*! \}
 */

} // namespace nih
