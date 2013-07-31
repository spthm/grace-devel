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

namespace nih {

template <uint32 DIM>
struct Bvh_builder<DIM>::Bvh_partitioner
{
	// constructor
	Bvh_partitioner(const uint32 split_dim, const float split_plane) :
		m_split_dim( split_dim ),
		m_split_plane( split_plane )
	{}

	// partitioner functor
	bool operator()(
		const typename Bvh_builder<DIM>::Point& v) const { return v.center(m_split_dim) < m_split_plane; }

	uint32 m_split_dim;
	float  m_split_plane;
};

template <uint32 DIM>
uint32 largest_dim(const Vector<float,DIM>& v)
{
	return uint32( std::max_element( &v[0], &v[0] + DIM ) - &v[0] );
}

void build_skip_nodes(
	Bvh_node*		root,
	Bvh_node*		node,
	const uint32	skip_node);

template <uint32 DIM>
template <typename Iterator>
void Bvh_builder<DIM>::build(
	const Iterator	begin,
	const Iterator	end,
	Bvh<DIM>*		bvh)
{
	m_points.resize( end - begin );
	uint32 i = 0;
	for (Iterator it = begin; it != end; ++it)
	{
		m_points[i].m_bbox  = *it;
		m_points[i].m_index = i;

		i++;
	}

	Node node;
	node.m_begin = 0u;
	node.m_end   = uint32(end - begin);
	node.m_depth = 0u;
	node.m_node  = 0u;

	Node_stack node_stack;
	node_stack.push( node );

	Bvh_node root;
	root.set_type( Bvh_node::kLeaf );
	root.set_index( 0u );
	bvh->m_nodes.push_back( root );
	bvh->m_bboxes.resize( 1u );

	while (node_stack.empty() == false)
	{
		const Node node = node_stack.top();
		node_stack.pop();
		
		const uint32 node_index = node.m_node;

		Bbox_type bbox;
		compute_bbox( node.m_begin, node.m_end, bbox );

		bvh->m_bboxes[ node_index ] = bbox;

		if (node.m_end - node.m_begin < m_max_leaf_size)
		{
			//
			// Make a leaf node
			//
			Bvh_node& bvh_node = bvh->m_nodes[ node_index ];

			bvh_node.set_type( Bvh_node::kLeaf );
			bvh_node.set_index( uint32( bvh->m_leaves.size() ) );
			bvh->m_leaves.push_back( Bvh_leaf( node.m_end - node.m_begin, node.m_begin ) );
		}
		else
		{
			//
			// Make a split node
			//

			// alloc space for children
			const uint32 left_node_index = uint32( bvh->m_nodes.size() );
			bvh->m_nodes.resize( left_node_index + 2 );
			bvh->m_bboxes.resize( left_node_index + 2 );

			Bvh_node& bvh_node = bvh->m_nodes[ node_index ];
			bvh_node.set_type( Bvh_node::kInternal );
			bvh_node.set_index( left_node_index );

			// find split plane
			const uint32 split_dim = largest_dim( bbox[1] - bbox[0] );
			const float split_plane = (bbox[1][split_dim] + bbox[0][split_dim]) * 0.5f;

			// partition the points
			Bvh_partitioner partitioner( split_dim, split_plane );
			uint32 middle = uint32(
				std::partition( &m_points[0] + node.m_begin, &m_points[0] + node.m_end, partitioner ) -
				&m_points[0] );

			// unsuccessful split: split in two equally sized subsets
			if (middle == node.m_begin ||
				middle == node.m_end)
				middle = (node.m_begin + node.m_end) / 2;

			// push left and right children in processing queue
			Node right_node;
			right_node.m_begin = middle;
			right_node.m_end   = node.m_end;
			right_node.m_depth = node.m_depth + 1u;
			right_node.m_node  = left_node_index + 1u;
			node_stack.push( right_node );

			Node left_node;
			left_node.m_begin = node.m_begin;
			left_node.m_end   = middle;
			left_node.m_depth = node.m_depth + 1u;
			left_node.m_node  = left_node_index;
			node_stack.push( left_node );
		}
	}

    build_skip_nodes( &bvh->m_nodes[0], &bvh->m_nodes[0], Bvh_node::kInvalid );
}

inline void build_skip_nodes(
	Bvh_node*		root,
	Bvh_node*		node,
	const uint32	skip_node)
{
	node->set_skip_node( skip_node );

	if (node->is_leaf() == false)
	{
		Bvh_node* l_node = root + node->get_index();
		Bvh_node* r_node = root + node->get_index() + 1u;

		build_skip_nodes( root, l_node, node->get_index() + 1u );
		build_skip_nodes( root, r_node, skip_node );
	}
}

template <uint32 DIM>
void Bvh_builder<DIM>::compute_bbox(
	const uint32		begin,
	const uint32		end,
	Bbox_type&			bbox)
{
	bbox.clear();
	for (uint32 i = begin; i < end; i++)
		bbox.insert( m_points[i].m_bbox );
}

inline Bvh_node::Bvh_node(const Type type, const uint32 index, const uint32 skip_node)
{
    m_packed_data = uint32( type ) | index;
    m_skip_node   = skip_node;
}

inline void Bvh_node::set_type(const Type type)
{
	m_packed_data &= ~kLeaf;
	m_packed_data |= uint32(type);
}
inline void Bvh_node::set_index(const uint32 index)
{
	m_packed_data &= kLeaf;
	m_packed_data |= index;
}
inline void Bvh_node::set_skip_node(const uint32 index)
{
	m_skip_node = index;
}

// compute SAH cost of a subtree
inline float compute_sah_cost(const Bvh<3>& bvh, uint32 node_index)
{
    const Bvh_node node = bvh.m_nodes[ node_index ];

    if (node.is_leaf())
    {
        const Bvh_leaf& leaf = bvh.m_leaves[ node.get_index() ];
        return float( leaf.get_size() );
    }
    else
    {
        const float cost1 = compute_sah_cost( bvh, node.get_index() );
        const float cost2 = compute_sah_cost( bvh, node.get_index()+1u );

        const Bbox3f bbox1 = bvh.m_bboxes[ node.get_index()    ];
        const Bbox3f bbox2 = bvh.m_bboxes[ node.get_index()+1u ];

        const Bbox3f bbox = bvh.m_bboxes[ node_index ];

        return 2.0f + (area( bbox1 ) * cost1 + area( bbox2 ) * cost2) / std::max( area( bbox ), 1.0e-6f );
    }
}

} // namespace nih