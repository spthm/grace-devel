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
#include <nih/basic/priority_queue.h>
#include <nih/bintree/bintree_node.h>
#include <nih/bintree/cuda/bintree_gen_context.h>
#include <thrust/device_vector.h>

namespace nih {
namespace cuda {

/*! \addtogroup kdtree k-d Trees
 */

/*! \addtogroup knn k-Nearest Neighbors
 *  \ingroup kdtree
 *  \{
 */

struct Kd_knn_result
{
    uint32 index;
    float  dist2;
};

///
/// GPU-based k-Nearest Neighbors lookup context.
///
/// This class provides the context to perform N-dimensional k-nn lookups on the GPU,
/// using a k-d tree.
///
template <uint32 DIM>
struct Kd_knn
{
    typedef Kd_knn_result Result;
};

///
/// GPU-based k-Nearest Neighbors lookup context.
///
/// This class provides the context to perform 3-dimensional k-nn lookups on the GPU,
/// using a k-d tree.
///
/// The following code snippet shows how to use this class:
///
/// \code
///
/// #include <nih/kd/cuda/kd_knn.h>
///
/// thrust::device_vector<Vector4f> kd_points;
/// thrust::device_vector<Kd_node>  kd_nodes;
/// thrust::device_vector<uint2>    kd_leaves;
/// thrust::device_vector<uint32>   kd_index;
///
/// thrust::device_vector<Vector4f> query_points;
/// ... // code to build the k-d tree and query points here...
///     // NOTE: even though we're doing 3-dimensional queries,
///     // we can use Vector4f arrays for better coalescing
///
/// thrust::device_vector<cuda::Kd_knn<3>::Result> results( query_points.size() );
///
/// nih::cuda::Kd_knn<3> knn;
/// knn.run(
///     query_points.begin(),
///     query_points.end(),
///     thrust::raw_pointer_cast( &kd_nodes.front() ),
///     thrust::raw_pointer_cast( &kd_ranges.front() ),
///     thrust::raw_pointer_cast( &kd_leaves.front() ),
///     thrust::raw_pointer_cast( &kd_points.front() ),
///     thrust::raw_pointer_cast( &results.front() ) );
/// 
///  \endcode
///
template <>
struct Kd_knn<3>
{
    typedef Kd_knn_result Result;

    /// perform a k-nn lookup for a set of query points
    ///
    /// \param points_begin     beginning of the query point sequence
    /// \param points_end       end of the query point sequence
    /// \param kd_nodes         k-d tree nodes
    /// \param kd_ranges        k-d tree node ranges
    /// \param kd_leaves        k-d tree leaves
    /// \param kd_points        k-d tree points
    ///
    /// \tparam QueryIterator   an iterator type dereferencing to Vector<float,N>, with N >= 3
    /// \tparam PointIterator   an iterator type dereferencing to Vector<float,N>, with N >= 3
    template <typename QueryIterator, typename PointIterator>
    void run(
        const QueryIterator             points_begin,
        const QueryIterator             points_end,
        const Kd_node*                  kd_nodes,
        const uint2*                    kd_ranges,
        const uint2*                    kd_leaves,
        const PointIterator             kd_points,
        Result*                         results);

    /// perform a k-nn lookup for a set of query points
    ///
    /// \param points_begin     beginning of the query point sequence
    /// \param points_end       end of the query point sequence
    /// \param kd_nodes         k-d tree nodes
    /// \param kd_ranges        k-d tree node ranges
    /// \param kd_leaves        k-d tree leaves
    /// \param kd_points        k-d tree points
    ///
    /// \tparam K               the number of requested nearest neighbors K
    /// \tparam QueryIterator   an iterator type dereferencing to Vector<float,N>, with N >= 3
    /// \tparam PointIterator   an iterator type dereferencing to Vector<float,N>, with N >= 3
    template <uint32 K, typename QueryIterator, typename PointIterator>
    void run(
        const QueryIterator             points_begin,
        const QueryIterator             points_end,
        const Kd_node*                  kd_nodes,
        const uint2*                    kd_ranges,
        const uint2*                    kd_leaves,
        const PointIterator             kd_points,
        Result*                         results);
};

/*! \}
 */

} // namespace cuda
} // namespace nih

#include <nih/kd/cuda/knn_inline.h>
