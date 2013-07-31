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

/*! \file kd_builder.h
 *   \brief Interface for a middle-split CUDA-based k-d tree builder.
 */

#pragma once

#include <nih/kd/kd_node.h>
#include <nih/linalg/vector.h>
#include <nih/linalg/bbox.h>
#include <nih/bintree/bintree_node.h>
#include <nih/bintree/cuda/bintree_gen_context.h>
#include <thrust/device_vector.h>

namespace nih {
namespace cuda {

/*! \addtogroup kdtree k-d Trees
 *  \{
 */

///
/// GPU-based middle-split k-d tree builder
///
/// This class provides the context to generate k-d trees on the GPU
/// starting from a set of unordered points.
/// The output is a set of nodes with the corresponding leaves and
/// a set of primitive indices into the input set of points.
/// The output leaves will specify contiguous ranges into this index.
///
/// \tparam Integer     an integer type that determines the number
///                     of bits used to compute the points' Morton codes.
///                     Accepted values are uint32 and uint64.
///
/// \tparam OutputTree  a template class used to write the output
///                     tree, with the following interface:
///
/// \code
/// struct OutputTree
/// {
///    void reserve_nodes(const uint32 n);  // reserve space for n nodes
///    void reserve_leaves(const uint32 n); // reserve space for n leaves
///
///    Context get_context();             // get a context to write nodes/leaves
///
///    struct Context
///    {
///        void write_node(
///           const uint32 node,          // node to write
///           const uint32 offset,        // child offset
///           const uint32 skip_node,     // skip node
///           const uint32 begin,         // node range begin
///           const uint32 end,           // node range end
///           const uint32 split_index,   // split index
///           const uint32 split_dim,     // splitting dimension
///           const uint32 split_plane);  // splitting plane
///
///        void write_node(
///           const uint32 node,          // node to write
///           const uint32 offset,        // child offset
///           const uint32 skip_node,     // skip node
///           const uint32 begin,         // node range begin
///           const uint32 end);          // node range end
///
///        void write_leaf(
///           const uint32 index,         // leaf to write
///           const uint32 begin,         // leaf range begin
///           const uint32 end);          // leaf range end
///    };
/// };
/// \endcode
///
/// The following code snippet shows how to use this builder:
///
/// \code
///
/// #include <nih/kd/cuda/kd_builder.h>
/// #include <nih/kd/cuda/kd_context.h>
///
/// thrust::device_vector<Vector3f> points;
/// ... // code to fill the input vector of points
///
/// thrust::device_vector<Kd_node>  kd_nodes;
/// thrust::device_vector<uint2>    kd_leaves;
/// thrust::device_vector<uint32>   kd_index;
///
/// nih::cuda::Kd_builder<uint64> builder( kd_index );
/// nih::cuda::Kd_context kd_tree( &kd_nodes, &kd_leaves, NULL );
/// builder.build(
///     kd_tree,                                    // output tree
///     kd_index,                                   // output index
///     Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),   // suppose all bboxes are in [0,1]^3
///     points.begin(),                             // begin iterator
///     points.end(),                               // end iterator
///     4 );                                        // target 4 objects per leaf
/// 
///  \endcode
///
template <typename Integer>
struct Kd_builder
{
    /// build a bvh given a set of points that will be reordered in-place
    ///
    /// \param out_tree         output tree
    /// \param out_index        output index
    /// \param bbox             global bbox
    /// \param points_begin     beginning of the point sequence to sort
    /// \param points_end       end of the point sequence to sort
    /// \param max_leaf_size    maximum leaf size
    template <typename OutputTree, typename Iterator>
    void build(
        OutputTree&                     out_tree,
        thrust::device_vector<uint32>&  out_index,
        const Bbox3f                    bbox,
        const Iterator                  points_begin,
        const Iterator                  points_end,
        const uint32                    max_leaf_size);

    thrust::device_vector<Integer>      m_codes;
    uint32                              m_levels[64];
    Bbox3f                              m_bbox;
    uint32                              m_node_count;
    uint32                              m_leaf_count;

    cuda::Bintree_gen_context           m_kd_context;
};

/*! \}
 */

} // namespace cuda
} // namespace nih

#include <nih/kd/cuda/kd_builder_inline.h>
