#include <thrust/device_vector.h>

#include "bintree_build_kernels.cuh"
#include "../nodes.h"
#include "bits.cuh"
#include "morton.cuh"


/* Instantiate the templates with types known to be used. */
namespace grace {

namespace gpu {

/* CUDA kernels. */

/* Build nodes with 32- or 64-bit keys. */
template __global__ void build_nodes_kernel<UInteger32>(Node *nodes,
                                                        Leaf *leaves,
                                                        UInteger32 *keys,
                                                        unsigned int n_keys,
                                                        unsigned char n_bits);
template __global__ void build_nodes_kernel<UInteger64>(Node *nodes,
                                                        Leaf *leaves,
                                                        UInteger64 *keys,
                                                        unsigned int n_keys,
                                                        unsigned char n_bits);

/* Find AABBs with positions represented as floats, doubles or long doubles. */
template __global__ void find_AABBs_kernel<float>(Node *nodes,
                                                  Leaf *leaves,
                                                  unsigned int n_leaves,
                                                  float *positions,
                                                  float *extent,
                                                  unsigned int *AABB_flags);
template __global__ void find_AABBs_kernel<double>(Node *nodes,
                                                   Leaf *leaves,
                                                   unsigned int n_leaves,
                                                   double *positions,
                                                   double *extent,
                                                   unsigned int *AABB_flags);
template __global__ void find_AABBs_kernel<long double>(Node *nodes,
                                                        Leaf *leaves,
                                                        unsigned int n_leaves,
                                                        long double *positions,
                                                        long double *extent,
                                                        unsigned int *AABB_flags);

template __device__ int common_prefix<UInteger32>(unsigned int i,
                                                  unsigned int j,
                                                  UInteger32 *keys,
                                                  unsigned int n_keys,
                                                  unsigned char n_bits);
template __device__ int common_prefix<UInteger64>(unsigned int i,
                                                  unsigned int j,
                                                  UInteger64 *keys,
                                                  unsigned int n_keys,
                                                  unsigned char n_bits);

} // namespace gpu

/* C-like wrappers for kernel calls. */

/* Build nodes with 32- or 64-bit keys. */
template void build_nodes<UInteger32>(thrust::device_vector<Node> d_nodes,
                                      thrust::device_vector<Leaf> d_leaves,
                                      thrust::device_vector<UInteger32> d_keys);
template void build_nodes<UInteger64>(thrust::device_vector<Node> d_nodes,
                                      thrust::device_vector<Leaf> d_leaves,
                                      thrust::device_vector<UInteger64> d_keys);

/* Find AABBs with positions represented as floats, doubles or long doubles. */
template void find_AABBs<float>(thrust::device_vector<Node> d_nodes,
                                       thrust::device_vector<Leaf> d_leaves,
                                       thrust::device_vector<float> d_sphere_centres,
                                       thrust::device_vector<float> d_sphere_radii);
template void find_AABBs<double>(thrust::device_vector<Node> d_nodes,
                                 thrust::device_vector<Leaf> d_leaves,
                                 thrust::device_vector<double> d_sphere_centres,
                                 thrust::device_vector<double> d_sphere_radii);
template void find_AABBs<long double>(thrust::device_vector<Node> d_nodes,
                                      thrust::device_vector<Leaf> d_leaves,
                                      thrust::device_vector<long double> d_sphere_centres,
                                      thrust::device_vector<long double> d_sphere_radii);

} // namespace grace
