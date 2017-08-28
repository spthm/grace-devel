#pragma once

#include "grace/execution_tag.h"
#include "grace/bvh.h"
#include "grace/detail/bvh_base.h"
#include "grace/cpp/detail/bvh_node.h"

#include <vector>

namespace grace {

namespace detail {

typedef BvhBase<std::vector<BvhNode>, std::vector<BvhLeaf> > HostBvhBase;

} // namespace detail


template <>
class Bvh<grace::cpp_tag> : public detail::HostBvhBase
{
public:
    GRACE_HOST
    explicit Bvh<grace::cpp_tag>(const size_t num_primitives,
                                 const int max_per_leaf = 1)
        : detail::HostBvhBase(num_primitives, max_per_leaf) {}
};

typedef Bvh<grace::cpp_tag> HostBvh;

} //namespace grace
