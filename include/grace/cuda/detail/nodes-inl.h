#pragma once

#include "grace/config.h"

#include "vector_config.h" // CUDA

namespace grace {

//-----------------------------------------------------------------------------
// Helper functions for tree build kernels.
//-----------------------------------------------------------------------------

GRACE_HOST_DEVICE bool is_empty_node::operator()(const int4 node) const
{
    // Note: a node's right child can never be node 0, and a leaf can never
    // cover zero elements.
    return (node.y == 0);
}

} //namespace grace
