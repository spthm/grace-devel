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

/*! \file octree_builder.h
 *   \brief Defines a CUDA octree builder
 */

#pragma once

#include <nih/octree/octree.h>
#include <nih/linalg/vector.h>
#include <nih/linalg/bbox.h>
#include <nih/bintree/bintree_node.h>
#include <nih/bintree/cuda/bintree_gen_context.h>
#include <thrust/device_vector.h>

namespace nih {

/*! \addtogroup octrees Octrees
 *  \{
 */

///
/// GPU-based octree builder
///
/// This class provides the context to generate octrees on the GPU
/// starting from a set of unordered points.
/// The output is a set of nodes with the corresponding leaves and
/// a set of primitive indices into the input set of points.
/// The output leaves will specify contiguous ranges into this index.
///
/// \tparam Integer     an integer type that determines the number
///                     of bits used to compute the points' Morton codes.
///                     Accepted values are uint32 and uint64.
///
/// The following code snippet shows how to use this builder:
///
/// \code
///
/// #include <nih/octree/cuda/octree_builder.h>
///
/// thrust::device_vector<Vector3f> points;
/// ... // code to fill the input vector of points
///
/// thrust::device_vector<Octree_node> octree_nodes;
/// thrust::device_vector<uint2>       octree_leaves;
/// thrust::device_vector<uint32>      octree_index;
///
/// nih::Octree_builder<uint64> builder( octree_nodes, octree_leaves, octree_index );
/// builder.build(
///     Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),   // suppose all bboxes are in [0,1]^3
///     points.begin(),                             // begin iterator
///     points.end(),                               // end iterator
///     4 );                                        // target 4 objects per leaf
/// 
///  \endcode
///
template <typename Integer>
struct Octree_builder
{
    /// constructor
    ///
    /// \param octree       output octree nodes array
    /// \param leaves       output leaf array
    /// \param index        output primitive index array
    Octree_builder(
        thrust::device_vector<Octree_node>&     octree,
        thrust::device_vector<uint2>&           leaves,
        thrust::device_vector<uint32>&          index) :
        m_octree( &octree ), m_leaves( &leaves ), m_index( &index ) {}

    /// build an octree given a set of points
    ///
    /// \param bbox             global bounding box
    /// \param points_begin     iterator to the beginning of the point sequence to sort
    /// \param points_end       iterator to the end of the point sequence to sort
    /// \param max_leaf_size    maximum leaf size
    template <typename Iterator>
    void build(
        const Bbox3f                           bbox,
        const Iterator                         points_begin,
        const Iterator                         points_end,
        const uint32                           max_leaf_size);

    thrust::device_vector<Octree_node>*      m_octree;
    thrust::device_vector<uint2>*            m_leaves;
    thrust::device_vector<uint32>*           m_index;
    thrust::device_vector<Integer>           m_codes;
    uint32                                   m_levels[64];
    Bbox3f                                   m_bbox;
    uint32                                   m_node_count;
    uint32                                   m_leaf_count;

    thrust::device_vector<Bintree_node>      m_kd_nodes;
    cuda::Bintree_gen_context                m_kd_context;
};

/*! \}
 */

} // namespace nih

#include <nih/octree/cuda/octree_builder_inline.h>