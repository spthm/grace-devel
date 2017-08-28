#pragma once

// This class exists only to provide access to a Bvh's private members
// without exposing them as part of the public API.
//
// DO NOT #include "bvh[_base].[cu]h" or #include "bvh-inl.[cu]h".
//
// For simplicity, bvh_base.h will include this file, and it will therefore
// always be available wherever a Bvh is used.

namespace grace {

namespace detail {

template <typename Bvh>
class Bvh_ref
{
public:
    typedef typename Bvh::node_vector node_vector;
    typedef typename Bvh::leaf_vector leaf_vector;
    typedef typename Bvh::node_vector::value_type node_type;
    typedef typename Bvh::leaf_vector::value_type leaf_type;
    typedef typename Bvh::node_vector::iterator node_iterator;
    typedef typename Bvh::leaf_vector::iterator leaf_iterator;

    GRACE_HOST Bvh_ref(Bvh& bvh) :
        nodes_ref_(bvh.nodes_), leaves_ref_(bvh.leaves_) {}

    node_vector& nodes()
    {
        return nodes_ref_;
    }

    const node_vector& nodes() const
    {
        return nodes_ref_;
    }

    leaf_vector& leaves()
    {
        return leaves_ref_;
    }


    const leaf_vector& leaves() const
    {
        return leaves_ref_;
    }

private:
    node_vector& nodes_ref_;
    leaf_vector& leaves_ref_;
};

template <typename Bvh>
class Bvh_const_ref
{
public:
    typedef typename Bvh::node_vector node_vector;
    typedef typename Bvh::leaf_vector leaf_vector;
    typedef typename Bvh::node_vector::value_type node_type;
    typedef typename Bvh::leaf_vector::value_type leaf_type;
    typedef typename Bvh::node_vector::iterator node_iterator;
    typedef typename Bvh::leaf_vector::iterator leaf_iterator;

    GRACE_HOST Bvh_const_ref(Bvh& bvh) :
        constnodes_ref_(bvh.nodes_), constleaves_ref_(bvh.leaves_) {}

    GRACE_HOST Bvh_const_ref(const Bvh& bvh) :
        constnodes_ref_(bvh.nodes_), constleaves_ref_(bvh.leaves_) {}

    node_vector& nodes()
    {
        return const_cast<node_vector&>(constnodes_ref_);
    }

    const node_vector& nodes() const
    {
        return constnodes_ref_;
    }

    leaf_vector& leaves()
    {
        return const_cast<leaf_vector&>(constleaves_ref_);
    }


    const leaf_vector& leaves() const
    {
        return constleaves_ref_;
    }

private:
    const node_vector& constnodes_ref_;
    const leaf_vector& constleaves_ref_;
};

} // namespace detail

} // namespace grace
