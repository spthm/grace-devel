#include <cassert>
#include "dvector.h"
#include "ray_compression.h"

// init skeleton --------------------------------------------------------
__global__ void init_skeleton(unsigned *skeleton, unsigned *head_flags,
                              int skel_value, size_t count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= count)
        return;

    skeleton[idx] = skel_value;
    head_flags[idx] = 0;
}

__host__ void init_skeleton(dvector<unsigned> &skel, 
                            dvector<unsigned> &head_flags,
                            int skel_value, size_t size)
{
    skel.resize(size);
    head_flags.resize(size);

    dim3 dimGrid, dimBlock;
    compute_linear_grid(size, dimGrid, dimBlock);

    init_skeleton<<<dimGrid, dimBlock>>>(skel.data(), head_flags.data(), 
                                         skel_value, size);
}


// create_num_packets --------------------------------------------------
__global__ void create_num_packets(const unsigned *chunk_size,
                                   unsigned *num_packets, size_t count,
                                   size_t max_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    num_packets[idx] = (chunk_size[idx] + max_size-1)/max_size;
}

__host__ void create_num_packets(dvector<unsigned> &num_packets,
                                 const dvector<unsigned> &chunk_size,
                                 size_t max_size)
{
    num_packets.resize(chunk_size.size());

    dim3 dimGrid, dimBlock;
    compute_linear_grid(chunk_size.size(), dimGrid, dimBlock);

    create_num_packets<<<dimGrid,dimBlock>>>(chunk_size, num_packets, 
                                             chunk_size.size(), max_size);
}


// decompose_into_packets --------------------------------------------

__global__ void calc_sizes(unsigned *sizes, const unsigned *positions,
                           const unsigned *num_positions,
                           const unsigned *comp_sizes, unsigned count, 
                           int max_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    unsigned pos = positions[idx];
    unsigned numpos = num_positions[idx];

    if(numpos > 1)
        sizes[pos] = max_size;

    int cnt = comp_sizes[idx] % max_size;
    sizes[pos+numpos-1] = cnt==0?max_size : cnt;
}

__global__ void decompose_compressed_rays(const unsigned *scan_num_packets,
                                          const unsigned *chunk_base,
                                          size_t count,
                                          unsigned *skeleton,
                                          unsigned *head_flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    int pos = scan_num_packets[idx];

    skeleton[pos] = chunk_base[idx];
    head_flags[pos] = 1;
}

__host__ void decompose_into_packets(dvector<unsigned> &packet_indices,
                                     dvector<unsigned> &packet_sizes,
                                     dvector<unsigned> *revmap,
                                     const dvector<unsigned> &comp_base,
                                     const dvector<unsigned> &comp_size,
                                     int max_size)
{
    // create num_packets array------------------------------------

    dvector<unsigned> num_packets;
    create_num_packets(num_packets, comp_size, max_size);

    // scan(num_packets) -----------------------------------------------

    dvector<unsigned> scan_num_packets;
    scan_add(scan_num_packets, num_packets, EXCLUSIVE);

    // get decomposition size -----------------------------------------

    unsigned decomp_count = scan_num_packets.back() + num_packets.back();

    // init skeleton for package ranges extraction ------------------

    dvector<unsigned> skeleton, head_flags;

    init_skeleton(skeleton, head_flags, max_size, decomp_count);

    // decompose compressed rays into packets --------------------

    dim3 dimGrid, dimBlock;
    compute_linear_grid(comp_base.size(), dimGrid, dimBlock);

    decompose_compressed_rays<<<dimGrid, dimBlock>>>(
        scan_num_packets, comp_base, comp_base.size(),
        skeleton, head_flags);

    segscan_add(packet_indices, skeleton, head_flags);

    calc_sizes<<<dimGrid, dimBlock>>>(skeleton, scan_num_packets,
                                      num_packets,
                                      comp_size, comp_size.size(), max_size);
    swap(packet_sizes, skeleton);

    if(revmap)
        scan_add(*revmap, head_flags, INCLUSIVE);

    assert(packet_sizes.size() == packet_indices.size());
}

__host__ void decompose_into_packets(dvector<unsigned> &packet_indices,
                                     dvector<unsigned> &packet_sizes,
                                     const dvector<unsigned> &ray_hashes,
                                     int max_size)
{
    dvector<unsigned> comp_hash, comp_base;
    compress_rays(comp_hash, comp_base, NULL, ray_hashes);

#if TRACE
    std::cout << "Recompressed size: " << comp_hash.size() << std::endl;
#endif

    dvector<unsigned> comp_size;
    adjacent_difference(comp_size, comp_base, ray_hashes.size());

    decompose_into_packets(packet_indices, packet_sizes, NULL,
                           comp_base, comp_size, max_size);
}


