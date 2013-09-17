#pragma once

namespace grace {

namesapce gpu {

template <typename UInteger>
__global__ void build_nodes_kernel(Node *nodes,
                                   Leaf *leaves,
                                   UInteger *keys,
                                   Uinteger n_keys,
                                   unsigned char n_bits);

template <typename UInteger, typename Float>
__global__ void find_AABBs_kernel(Node *nodes
                                  Leaf *leaves
                                  UInteger n_leaves,
                                  Float *positions,
                                  Float *extent,
                                  unsigned int *AABB_flags);

} // namespace gpu

template <typename UInteger>
__host__ __device__ unsigned int common_prefix(UInteger i,
                                               UInteger j,
                                               UInteger *keys,
                                               UInteger n_keys,
                                               unsigned char n_bits);


} // namespace grace
