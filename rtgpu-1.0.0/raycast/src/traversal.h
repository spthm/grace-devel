#ifndef TRAVERSAL_H
#define TRAVERSAL_H

template <class T> class dvector;
struct bvh_soa;

template <class F>
__host__ void traverse(dvector<unsigned> &d_active_ray_packets,
                       dvector<unsigned> &d_active_ray_packet_sizes,
                       dvector<unsigned> &d_active_frustum_leaves,
                       dvector<unsigned> &d_active_frustum_leaf_sizes,
                       dvector<unsigned> &d_active_idleaves,
                       unsigned root_children_count, const bvh_soa &d_bvh,
                       unsigned bvh_height,
                       const F &d_frustums,
                       const dvector<unsigned> &d_ray_packets,
                       const dvector<unsigned> &d_ray_packet_sizes);

#endif
