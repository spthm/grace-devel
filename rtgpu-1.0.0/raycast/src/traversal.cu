#include <cassert>
#include "traversal.h"
#include <iostream>
#include "frustum.h"
#include "ray_compression.h"
#include "cuda_bvh.h"
#include "aabb.h"
#include "timer.h"

#define MAX_DEPTH 0

texture<float4> tex_bvh_node_aabb1;
texture<float2> tex_bvh_node_aabb2;
texture<unsigned> tex_bvh_node_children_info;

inline __device__ AABB bvh_node_aabb(int idnode)
{
    return make_aabb(tex1Dfetch(tex_bvh_node_aabb1, idnode),
                     tex1Dfetch(tex_bvh_node_aabb2, idnode));
}

//{{{ intersect frustum with nodes -----------------------------------

template <class F>
__global__ void intersect_root(unsigned root_children_count,/*{{{*/
                          const F frustums,
                          unsigned *in_frustum_ids, 
                          unsigned *in_node_ids, 
                          unsigned *frustum_intersection_count,
                          unsigned *intersected_children,
                          unsigned *reserved_space=NULL)
{
    int idfrustum = blockIdx.x*blockDim.x + threadIdx.x;

    if(idfrustum >= frustums.size)
        return;

    in_frustum_ids[idfrustum] = idfrustum;
    in_node_ids[idfrustum] = 0;

    Frustum frustum;
    frustum.top = frustums.top[idfrustum];
    frustum.right = frustums.right[idfrustum];
    frustum.bottom = frustums.bottom[idfrustum];
    frustum.left = frustums.left[idfrustum];
    frustum.dirsign = frustums.dirsign[idfrustum];

    unsigned num_intersections = 0;
    unsigned sorted_children = 0;

    for(int i=0; i<8; ++i)
    {
        unsigned idchild = i ^ frustum.dirsign;

        if(idchild < root_children_count &&
            intersects(bvh_node_aabb(1+idchild), frustum))
        {
            sorted_children |= idchild << num_intersections*3;
            ++num_intersections;
        }
    }

    frustum_intersection_count[idfrustum] = num_intersections;
    intersected_children[idfrustum] = sorted_children;
}/*}}}*/

template <class F>
__global__ void intersect(const F frustums,/*{{{*/
                          unsigned *in_frustum_ids, 
                          unsigned *in_node_ids, 
                          unsigned count,
                          unsigned *frustum_intersection_count,
                          unsigned *intersected_children,
                          unsigned *reserved_space=NULL)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    int idfrustum, idnode;

    idfrustum = in_frustum_ids[idx];
    idnode    = in_node_ids[idx];

    unsigned children_info = tex1Dfetch(tex_bvh_node_children_info, idnode);

    char children_count = children_info&0xFF;
    unsigned children_offset = children_info>>8;

    // usar textura p/ acessar frustum deixa (muito) mais lento
    Frustum frustum;
    frustum.top = frustums.top[idfrustum];
    frustum.right = frustums.right[idfrustum];
    frustum.bottom = frustums.bottom[idfrustum];
    frustum.left = frustums.left[idfrustum];
    frustum.dirsign = frustums.dirsign[idfrustum];

    unsigned num_intersections = 0;
    unsigned sorted_children = 0;

    for(int i=0; i<8; ++i)
    {
        unsigned idchild = i ^ frustum.dirsign;

        if(idchild < children_count &&
           intersects(bvh_node_aabb(children_offset+idchild), frustum))
        {
            sorted_children |= idchild << num_intersections*3;
            ++num_intersections;
        }
    }

    frustum_intersection_count[idx] = num_intersections;
    intersected_children[idx] = sorted_children;
//    if(reserved_space)
//        reserved_space[idx] = (num_intersections+3)&~3;
}/*}}}*/

template <class F>
__host__ void intersect(unsigned root_children_count,
                        const F &frustums,
                        dvector<unsigned> &in_frustum_ids, 
                        dvector<unsigned> &in_node_ids, 
                        dvector<unsigned> &frustum_intersection_count, 
                        dvector<unsigned> &intersected_children, 
                        dvector<unsigned> *reserved_children_space,
                        int depth)
{
    assert(in_frustum_ids.size() == in_node_ids.size());

    if(depth == 0)
    {
        in_frustum_ids.resize(frustums.size());
        in_node_ids.resize(frustums.size());
    }

    frustum_intersection_count.resize(in_frustum_ids.size());
    intersected_children.resize(in_frustum_ids.size());

    if(reserved_children_space)
        reserved_children_space->resize(in_frustum_ids.size());

    dim3 dimBlock(64);
    dim3 dimGrid((in_frustum_ids.size()+dimBlock.x-1)/dimBlock.x);

    const_FrustumsGPU fgpu = frustums;

#if MAX_DEPTH
    if(depth+1 == MAX_DEPTH)
    {
        cuda_timer timer;

        if(depth==0)
        {
            intersect_root<<<dimGrid, dimBlock>>>(
                root_children_count, fgpu, in_frustum_ids, in_node_ids, 
                frustum_intersection_count, intersected_children);

            check_cuda_error("intersect_root kernel");
        }
        else
        {
            intersect<<<dimGrid, dimBlock>>>(
                fgpu, in_frustum_ids, in_node_ids, in_frustum_ids.size(),
                frustum_intersection_count, intersected_children);
            check_cuda_error("intersect kernel");
        }

        timer.stop();

#if 1
        std::vector<unsigned> 
            h_num_intersected = to_cpu(frustum_intersection_count),
            h_intersected_children = to_cpu(intersected_children),
            h_in_frustum_ids = to_cpu(in_frustum_ids),
            h_in_node_ids = to_cpu(in_node_ids);

        for(int j=0; j<h_num_intersected.size(); ++j)
        {
        //    if(h_num_intersected[j])
                std::cout  << h_in_frustum_ids[j] << "," << h_in_node_ids[j] << ": " << h_num_intersected[j] << " - " << h_intersected_children[j] << std::endl;
        }
#endif
        std::cout << "Elapsed: " << timer.elapsed() << std::endl;

        exit(0);
    }
    else
#endif
    if(depth==0)
    {
        intersect_root<<<dimGrid, dimBlock>>>(
            root_children_count, fgpu, in_frustum_ids, in_node_ids,
            frustum_intersection_count, intersected_children, 
            reserved_children_space?reserved_children_space->data():NULL);

        check_cuda_error("intersect_root kernel");
    }
    else
    {
        intersect<<<dimGrid, dimBlock>>>(
            fgpu, in_frustum_ids, in_node_ids, in_frustum_ids.size(),
            frustum_intersection_count, intersected_children,
            reserved_children_space?reserved_children_space->data():NULL);

        check_cuda_error("intersect kernel");
    }

}/*}}}*/

// scatter frustums ---------------------------------------------

__global__ void gather_frustums(unsigned *out_indices, 
                                unsigned *out_sizes,
                                const unsigned *in_indices, 
                                const unsigned *in_sizes,
                                const unsigned *frustums,
                                unsigned count)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    unsigned idfrustum = frustums[idx];

    out_indices[idx] = in_indices[idfrustum];
    out_sizes[idx] = in_sizes[idfrustum];
}

__host__ void gather_frustums(dvector<unsigned> &out_indices,
                              dvector<unsigned> &out_sizes,
                              const dvector<unsigned> &in_indices,
                              const dvector<unsigned> &in_sizes,
                              const dvector<unsigned> &frustums)
{
    out_indices.resize(frustums.size());
    out_sizes.resize(frustums.size());

    dim3 dimGrid, dimBlock;
    compute_linear_grid(frustums.size(), dimGrid, dimBlock);

    gather_frustums<<<dimGrid, dimBlock>>>(out_indices, out_sizes,
                                           in_indices, in_sizes,
                                           frustums, frustums.size());

    check_cuda_error("gather_frustums kernel");
}

__global__ 
void scatter_frustums(unsigned *out_frustum_ids, 
                      unsigned *out_node_ids, 
                      const unsigned *in_frustum_ids, 
                      const unsigned *in_node_ids,
                      const unsigned *num_intersected, 
                      const unsigned *dest_positions,
                      const unsigned *intersected_children,
                      unsigned count)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    unsigned idfrustum = in_frustum_ids[idx];
    int dest_idx = dest_positions[idx];
    unsigned children_offset = tex1Dfetch(tex_bvh_node_children_info,
                                          in_node_ids[idx])>>8;
    unsigned intersected = intersected_children[idx];
    unsigned num_inter = num_intersected[idx];

    for(int i=0; i<num_inter; ++i)
    {
        out_frustum_ids[dest_idx+i] = idfrustum;
        out_node_ids[dest_idx+i] = children_offset+(intersected & 0x7);
        intersected >>= 3;
    }
}

__host__ 
void scatter_frustums(dvector<unsigned> &out_frustum_ids, 
               dvector<unsigned> &out_node_ids, 
               const dvector<unsigned> &in_frustum_ids, 
               const dvector<unsigned> &in_node_ids,
               const dvector<unsigned> &num_intersected, 
               const dvector<unsigned> &dest_positions,
               unsigned out_size,
               const dvector<unsigned> &intersected_children)
{
    assert(in_frustum_ids.size() == in_node_ids.size());
    assert(in_node_ids.size() == num_intersected.size());
    assert(num_intersected.size() == dest_positions.size());
    assert(dest_positions.size() == intersected_children.size());

    out_frustum_ids.resize(out_size);
    out_node_ids.resize(out_size);

    dim3 dimGrid, dimBlock;
    compute_linear_grid(in_frustum_ids.size(), dimGrid, dimBlock);

    scatter_frustums<<<dimGrid, dimBlock>>>(out_frustum_ids, out_node_ids,
                                            in_frustum_ids, in_node_ids,
                                            num_intersected, 
                                            dest_positions,
                                            intersected_children,
                                            in_frustum_ids.size());

    check_cuda_error("scatter_frustums kernel");
}

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
                       const dvector<unsigned> &d_ray_packet_sizes)
{

    static dvector<unsigned> d_in_frustum_ids, d_in_node_ids,
                             d_out_frustum_ids,
                             d_num_intersected,
                             d_dest_positions,
                             d_intersected_children,
                             d_out_node_ids;

    cudaBindTexture(NULL, tex_bvh_node_aabb1, d_bvh.aabb1);
    check_cuda_error("Binding aabb1 to texture");

    cudaBindTexture(NULL, tex_bvh_node_aabb2, d_bvh.aabb2);
    check_cuda_error("Binding aabb2 to texture");

    cudaBindTexture(NULL, tex_bvh_node_children_info, d_bvh.children_info);
    check_cuda_error("Binding children_info to texture");

    scoped_timer_stop sts(timers.add("BVH traversal"));

    // first and intermediate levels
    for(int i=0; i<bvh_height-1; ++i)
    {
        intersect(root_children_count,
                  d_frustums, d_in_frustum_ids, d_in_node_ids, 
                  d_num_intersected, d_intersected_children, NULL, i);

        scan_add(d_dest_positions, d_num_intersected, EXCLUSIVE);


        scatter_frustums(d_out_frustum_ids, d_out_node_ids, 
                         d_in_frustum_ids, d_in_node_ids,
                         d_num_intersected, d_dest_positions,
                         d_dest_positions.back()+d_num_intersected.back(),
                         d_intersected_children);

        swap(d_in_frustum_ids, d_out_frustum_ids);
        swap(d_in_node_ids, d_out_node_ids);
    }

    // last level
    static dvector<unsigned> d_reserved_children_space;

    intersect(root_children_count, 
              d_frustums, d_in_frustum_ids, d_in_node_ids, 
              d_num_intersected, d_intersected_children,
              NULL, bvh_height-1);

    scan_add(d_dest_positions, d_num_intersected, EXCLUSIVE);

    scatter_frustums(d_out_frustum_ids, d_out_node_ids,
                     d_in_frustum_ids, d_in_node_ids,
                     d_num_intersected, d_dest_positions,
                     d_dest_positions.back()+d_num_intersected.back(),
                     d_intersected_children);

    swap(d_out_node_ids, d_active_idleaves);

    static dvector<unsigned> d_active_idfrustums;

    compress_rays(d_active_idfrustums, d_active_frustum_leaves, NULL,
                  d_out_frustum_ids);

    adjacent_difference(d_active_frustum_leaf_sizes, 
                        d_active_frustum_leaves, 
                        d_active_idleaves.size());

    gather_frustums(d_active_ray_packets, d_active_ray_packet_sizes,
                    d_ray_packets, d_ray_packet_sizes,
                    d_active_idfrustums);

    assert(d_active_frustum_leaf_sizes.size()==d_active_frustum_leaves.size());
    assert(d_active_frustum_leaves.size() == d_active_idfrustums.size());
}

template
void traverse(dvector<unsigned> &d_active_ray_packets,
                       dvector<unsigned> &d_active_ray_packet_sizes,
                       dvector<unsigned> &d_active_frustum_leaves,
                       dvector<unsigned> &d_active_frustum_leaf_sizes,
                       dvector<unsigned> &d_active_idleaves,
                       unsigned root_children_count, const bvh_soa &d_bvh,
                       unsigned bvh_height,
                       const Frustums &d_frustums,
                       const dvector<unsigned> &d_ray_packets,
                       const dvector<unsigned> &d_ray_packet_sizes);
