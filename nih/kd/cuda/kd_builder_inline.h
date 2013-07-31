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

#include <nih/bintree/cuda/bintree_gen.h>
#include <nih/bits/morton.h>
#include <thrust/sort.h>

namespace nih {
namespace cuda {

namespace kd {

    template <typename Integer>
    struct Morton_bits {};

    template <>
    struct Morton_bits<uint32> { static const uint32 value = 30u; };

    template <>
    struct Morton_bits<uint64> { static const uint32 value = 60u; };

    NIH_HOST_DEVICE inline float convert(float a, float b, const uint32 i)
    {
        const float x = float(i) / float(1u << 10u);
        return a + (b - a) * x;
    }
    NIH_HOST_DEVICE inline float convert(float a, float b, const uint64 i)
    {
        const float x = float(i) / float(1u << 20u);
        return a + (b - a) * x;
    }

/// A simple binary tree context implementation to be used with
/// the Bvh generate() function.
template <typename Integer, typename OutputTree>
struct Kd_context
{
    typedef typename OutputTree::Context BaseContext;

    /// Cuda accessor struct
    struct Context
    {
        NIH_HOST_DEVICE Context() {}
        NIH_HOST_DEVICE Context(const BaseContext context, const Integer* codes, Bbox3f bbox) :
            m_context( context ), m_codes( codes ), m_bbox( bbox ) {}

        /// write a new node
        NIH_HOST_DEVICE void write_node(const uint32 node, bool p1, bool p2, const uint32 offset, const uint32 skip_node, const uint32 level, const uint32 begin, const uint32 end, const uint32 split_index)
        {
            if (p1)
            {
                // fetch the Morton code corresponding to the split plane
                      Integer code = m_codes[ split_index ];
                const uint32  split_dim = level % 3;

                // extract the selected coordinate
                Integer split_coord = 0;

                code >>= level-1;
                code <<= level-1;

                for (int i = 0; code; i++)
                {
	                split_coord |= (((code >> split_dim) & 1u) << i);
                    code >>= 3u;
                }

                // convert to floating point
                const float split_plane = convert( m_bbox[0][split_dim], m_bbox[1][split_dim], split_coord );

                // and output the split node
                m_context.write_node(
                    node,
                    offset,
                    skip_node,
                    begin,
                    end,
                    split_index,
                    split_dim,
                    split_plane );
            }
            else
            {
                // output a leaf node
                m_context.write_node(
                    node,
                    offset,
                    skip_node,
                    begin,
                    end );
            }
        }
        /// write a new leaf
        NIH_HOST_DEVICE void write_leaf(const uint32 index, const uint32 begin, const uint32 end)
        {
            m_context.write_leaf( index, begin, end );
        }

        BaseContext     m_context;
        const Integer*  m_codes;
        Bbox3f          m_bbox;
    };

    /// constructor
    Kd_context(
        OutputTree                   context,
        const Integer*                  codes,
        Bbox3f                          bbox) :
        m_context( context ), m_codes( codes ), m_bbox( bbox ) {}

    /// reserve space for more nodes
    void reserve_nodes(const uint32 n) { m_context.reserve_nodes(n); }

    /// reserve space for more leaves
    void reserve_leaves(const uint32 n) { m_context.reserve_leaves(n); }

    /// return a cuda context
    Context get_context()
    {
        return Context(
            m_context.get_context(),
            m_codes,
            m_bbox );
    }

    OutputTree                    m_context;
    const Integer*                   m_codes;
    Bbox3f                           m_bbox;
};

};

// build a k-d tree given a set of points
template <typename Integer>
template <typename OutputTree, typename Iterator>
void Kd_builder<Integer>::build(
    OutputTree&                     tree,
    thrust::device_vector<uint32>&  index,
    const Bbox3f                    bbox,
    const Iterator                  points_begin,
    const Iterator                  points_end,
    const uint32                    max_leaf_size)
{ 
    typedef cuda::Bintree_gen_context::Split_task Split_task;

    const uint32 n_points = uint32( points_end - points_begin );

    m_bbox = bbox;
    need_space( m_codes, n_points );
    need_space( index, n_points );

    // compute the Morton code for each point
    thrust::transform(
        points_begin,
        points_begin + n_points,
        m_codes.begin(),
        morton_functor<Integer>( bbox ) );

    // setup the point indices, from 0 to n_points-1
    thrust::copy(
        thrust::counting_iterator<uint32>(0),
        thrust::counting_iterator<uint32>(0) + n_points,
        index.begin() );

    // sort the indices by Morton code
    // TODO: use Duane's library directly here... this is doing shameful allocations!
    thrust::sort_by_key(
        m_codes.begin(),
        m_codes.begin() + n_points,
        index.begin() );

    // generate a kd-tree
    kd::Kd_context<Integer,OutputTree> bintree_context( tree, thrust::raw_pointer_cast( &m_codes.front() ), m_bbox );

    const uint32 bits = kd::Morton_bits<Integer>::value;

    generate(
        m_kd_context,
        n_points,
        thrust::raw_pointer_cast( &m_codes.front() ),
        bits,
        max_leaf_size,
        false,
        bintree_context );

    m_leaf_count = m_kd_context.m_leaves;
    m_node_count = m_kd_context.m_nodes;

    for (uint32 level = 0; level <= bits; ++level)
        m_levels[ bits - level ] = m_kd_context.m_levels[ level ];
}

} // namespace cuda
} // namespace nih
