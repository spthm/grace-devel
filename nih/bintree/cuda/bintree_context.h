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

/*! \file bintree_context.h
 *   \brief Defines a simple binary tree context implementation to be used with
 *          the Bintree generate() function.
 */

#pragma once

#include <nih/basic/types.h>
#include <nih/bintree/bintree_node.h>
#include <thrust/device_vector.h>

namespace nih {
namespace cuda {

/*! \addtogroup bintree Binary Trees
 *  \{
 */

/// A simple binary tree context implementation to be used with
/// the Bintree generate() function.
struct Bintree_context
{
    /// Cuda accessor struct
    ///
    struct Context
    {
        /// empty constructor
        ///
        NIH_HOST_DEVICE Context() {}
        /// constructor
        ///
        NIH_HOST_DEVICE Context(Bintree_node* nodes, uint2* leaves) :
            m_nodes(nodes), m_leaves(leaves) {}

        /// write a new node
        ///
        NIH_HOST_DEVICE void write_node(const uint32 node, bool p1, bool p2,
                                        const uint32 offset, const uint32 skip_node,
                                        const uint32 level, const uint32 begin,
                                        const uint32 end, const uint32 split_index)
        {
            m_nodes[ node ] = Bintree_node( p1, p2, offset );
        }
        /// write a new leaf
        ///
        NIH_HOST_DEVICE void write_leaf(const uint32 index, const uint32 begin,
                                        const uint32 end)
        {
            m_leaves[ index ] = make_uint2( begin, end );
        }

        Bintree_node*  m_nodes;    ///< node pointer
        uint2*         m_leaves;   ///< leaf pointer
    };

    /// constructor
    ///
    /// \param nodes    nodes to write to
    /// \param leaves   leaves to write to
    Bintree_context(
        thrust::device_vector<Bintree_node>& nodes,
        thrust::device_vector<uint2>&        leaves) :
        m_nodes( &nodes ), m_leaves( &leaves ) {}

    /// constructor
    ///
    /// \param nodes    nodes to write to
    /// \param leaves   leaves to write to
    Bintree_context(
        thrust::device_vector<Bintree_node>* nodes,
        thrust::device_vector<uint2>*        leaves) :
        m_nodes( nodes ), m_leaves( leaves ) {}

    /// reserve space for more nodes
    ///
    /// \param n    nodes to reserve
    void reserve_nodes(const uint32 n) { if (m_nodes->size() < n) m_nodes->resize(n); }

    /// reserve space for more leaves
    ///
    /// \param n    leaves to reserve
    void reserve_leaves(const uint32 n) { if (m_leaves->size() < n) m_leaves->resize(n); }

    /// return a cuda context
    ///
    Context get_context()
    {
        return Context(
            thrust::raw_pointer_cast( &m_nodes->front() ),
            thrust::raw_pointer_cast( &m_leaves->front() ) );
    }

    thrust::device_vector<Bintree_node>* m_nodes;
    thrust::device_vector<uint2>*        m_leaves;
};

/*! \}
 */

} // namespace cuda
} // namespace nih
