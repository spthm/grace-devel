#ifndef LOCAL_INTERSECTIONS_H
#define LOCAL_INTERSECTIONS_H

struct bvh_soa;

template <class T> class dvector;
struct linear_8bvh_leaf;

__host__ void primary_local_intersections(
    float4 *output,

    const dvector<unsigned> &d_active_ray_packets,
    const dvector<unsigned> &d_active_ray_packet_sizes,
    const dvector<unsigned> &d_active_frustum_leaves,
    const dvector<unsigned> &d_active_frustum_leaf_sizes,
    const dvector<unsigned> &d_active_idleaves,
    const bvh_soa &d_bvh,

    const dvector<float4> &d_tri_xform,
    const dvector<float4> &d_tri_normals,

    float3 ray_ori, const dvector<float3> &d_rays_dir,
    const dvector<unsigned> &d_rays_idx);

__host__ void shadow_local_intersections(
    dvector<unsigned> &d_on_shadow,

    const dvector<unsigned> &d_active_ray_packets,
    const dvector<unsigned> &d_active_ray_packet_sizes,
    const dvector<unsigned> &d_active_frustum_leaves,
    const dvector<unsigned> &d_active_frustum_leaf_sizes,
    const dvector<linear_8bvh_leaf> &d_active_leaves,

    const dvector<float4> &d_tri_xform,

    const dvector<float3> &d_ray_oris, const dvector<float3> &d_rays_dir,
    const dvector<unsigned> &d_rays_idx);

#endif
