#pragma once

#include "../nodes.h"

namespace grace {

namespace gpu {

template <typename UInteger>
__global__ void build_nodes_kernel(Node *nodes,
                                   Leaf *leaves,
                                   UInteger *keys,
                                   unsigned int n_keys,
                                   unsigned char n_bits);

template <typename Float>
__global__ void find_AABBs_kernel(Node *nodes,
                                  Leaf *leaves,
                                  unsigned int n_leaves,
                                  Float *positions,
                                  Float *extent,
                                  unsigned int *AABB_flags);

template <typename UInteger>
__device__ int common_prefix(unsigned int i,
                                      unsigned int j,
                                      UInteger *keys,
                                      unsigned int n_keys,
                                      unsigned char n_bits);

} // namespace gpu

template <typename UInteger>
void build_nodes(thrust::device_vector<Node> d_nodes,
                 thrust::device_vector<Leaf> d_leaves,
                 thrust::device_vector<UInteger> d_keys);

template <typename Float>
void find_AABBs(thrust::device_vector<Node> d_nodes,
                thrust::device_vector<Leaf> d_leaves,
                thrust::device_vector<Float> d_sphere_centres,
                thrust::device_vector<Float> d_sphere_radii);


} // namespace grace
