#include "ray_compression.h"
#include "dvector.h"
#include "timer.h"

// compress rays ---------------------------------------------------------
__global__ void compress_rays(const unsigned *ray_hashes,
                              const unsigned *head_flags,
                              const unsigned *scan_head_flags,
                              size_t count,
                              unsigned *chunk_hash,
                              unsigned *chunk_base,
                              unsigned *chunk_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    if(head_flags[idx])
    {
        int cidx = scan_head_flags[idx];
        chunk_hash[cidx] = ray_hashes[idx];
        chunk_base[cidx] = idx;
        if(chunk_idx)
            chunk_idx[cidx] = cidx;
    }
}

__global__ void calc_head_flags(const unsigned *hashes, unsigned *head_flags, 
                                size_t count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    if(idx == 0)
        head_flags[0] = 1;
    else
        head_flags[idx] = hashes[idx] != hashes[idx-1] ? 1 : 0;
}

__host__ void compress_rays(dvector<unsigned> &chunk_hash,
                            dvector<unsigned> &chunk_base,
                            dvector<unsigned> *chunk_idx,
                            const dvector<unsigned> &ray_hashes)
{
    size_t ray_count = ray_hashes.size();

    dim3 dimGrid, dimBlock;
    compute_linear_grid(ray_count, dimGrid, dimBlock);

    dvector<unsigned> head_flags(ray_count);

    // create head flags ------------------------------------------

    cuda_timer &t1 = timers.add(" -- head flags creation");
    calc_head_flags<<<dimGrid, dimBlock>>>(ray_hashes, head_flags, ray_count);
    t1.stop();

    // create scan(head flags) ------------------------------------

    cuda_timer &t2 = timers.add(" -- scan add");
    dvector<unsigned> scan_head_flags;
    scan_add(scan_head_flags, head_flags, EXCLUSIVE);
    t2.stop();

    // get compressed size-----------------------------------------

    unsigned comp_count = scan_head_flags.back() + head_flags.back();

    // compress and sort rays -------------------------------------------

    chunk_base.resize(comp_count);
    chunk_hash.resize(comp_count);
    if(chunk_idx)
        chunk_idx->resize(comp_count);

    cuda_timer &t3 = timers.add(" -- compression");
    compress_rays<<<dimGrid, dimBlock>>>(
        ray_hashes, head_flags, scan_head_flags, ray_count, 
        chunk_hash, chunk_base, chunk_idx?chunk_idx->data():NULL);
    t3.stop();
}

