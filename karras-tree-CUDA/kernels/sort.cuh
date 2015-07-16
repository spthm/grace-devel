#pragma once

#include "../types.h"
#include "../utils.cuh"

// moderngpu/include must be in the INC path
#include "kernels/segmentedsort.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <thrust/iterator/constant_iterator.h>

#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace grace {

GRACE_HOST void offsets_to_segments(
    const thrust::device_vector<int>& d_offsets,
    thrust::device_vector<int>& d_segments)
{
    size_t N_offsets = d_offsets.size();
    thrust::constant_iterator<unsigned int> first(1);
    thrust::constant_iterator<unsigned int> last = first + N_offsets;

    // Suppose offsets = [0, 3, 3, 7]
    // scatter value 1 at offsets[1:N] into segments:
    //    => ray_segments = [0, 0, 0, 1, 0, 0, 0, 1(, 0 ... )]
    // inclusive_scan:
    //    => ray_segments = [0, 0, 0, 1, 1, 1, 1, 2(, 2 ... )]
    // Note that we do not scatter a 1 into ray_segments[offsets[0]].
    thrust::scatter(first+1, last,
                    d_offsets.begin()+1,
                    d_segments.begin());
    thrust::inclusive_scan(d_segments.begin(), d_segments.end(),
                           d_segments.begin());
}

template <typename UInteger, typename T>
GRACE_HOST void order_by_index(
    const thrust::device_vector<UInteger>& d_indices,
    thrust::device_vector<T>& d_unordered)
{
    thrust::device_vector<T> d_tmp = d_unordered;
    thrust::gather(d_indices.begin(), d_indices.end(),
                   d_tmp.begin(),
                   d_unordered.begin());
}

template <typename T, typename UInteger>
GRACE_HOST void sort_and_map(
    thrust::device_vector<T>& d_unsorted,
    thrust::device_vector<UInteger>& d_map)
{
    thrust::sequence(d_map.begin(), d_map.end(), 0u);
    thrust::sort_by_key(d_unsorted.begin(), d_unsorted.end(), d_map.begin());
}

// Like sort_and_map, but does not touch the original, unsorted vector.
template <typename T, typename UInteger>
GRACE_HOST void sort_map(
    thrust::device_vector<T>& d_unsorted,
    thrust::device_vector<UInteger>& d_map)
{
    thrust::sequence(d_map.begin(), d_map.end(), 0u);
    thrust::device_vector<T> d_tmp = d_unsorted;
    thrust::sort_by_key(d_tmp.begin(), d_tmp.end(), d_map.begin());
}

template <typename T_key, typename Ta, typename Tb>
GRACE_HOST void sort_by_key(
    thrust::host_vector<T_key>& h_keys,
    thrust::host_vector<Ta>& h_a,
    thrust::host_vector<Tb>& h_b)
{
    thrust::host_vector<T_key> h_keys2 = h_keys;

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_a.begin());

    thrust::sort_by_key(h_keys2.begin(), h_keys2.end(), h_b.begin());
}

template <typename T_key, typename Ta, typename Tb>
GRACE_HOST void sort_by_key(
    thrust::device_vector<T_key>& d_keys,
    thrust::device_vector<Ta>& d_a,
    thrust::device_vector<Tb>& d_b)
{
    thrust::device_vector<T_key> d_keys2 = d_keys;

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_a.begin());

    thrust::sort_by_key(d_keys2.begin(), d_keys2.end(), d_b.begin());
}

// MGPU's KernelSegBlocksortIndices, modified to be tolerant to grid sizes which
// exceed the device maximum.
template<typename Tuning, bool Stable, bool HasValues, typename InputIt1,
         typename InputIt2, typename OutputIt1, typename OutputIt2,
         typename Comp>
MGPU_LAUNCH_BOUNDS void KernelSegBlocksortIndices(
    InputIt1 keys_global,
    InputIt2 values_global,
    int count,
    int numBlocks,
    const int* indices_global,
    const int* partitions_global,
    OutputIt1 keysDest_global,
    OutputIt2 valsDest_global,
    int* ranges_global,
    Comp comp) {

    typedef typename std::iterator_traits<InputIt1>::value_type KeyType;
    typedef typename std::iterator_traits<InputIt2>::value_type ValType;
    typedef MGPU_LAUNCH_PARAMS Params;
    const int NT = Params::NT;
    const int VT = Params::VT;
    const int NV = NT * VT;

    const int FlagWordsPerThread = MGPU_DIV_UP(VT, 4);
    struct Shared {
        union {
            mgpu::byte flags[NV];
            int words[NT * FlagWordsPerThread];
            KeyType keys[NV];
            ValType values[NV];
        };
        int ranges[NT];
    };
    __shared__ Shared shared;

    int tid = threadIdx.x;
    int block = blockIdx.x;

    // Compute capability 3.0+ devices have x-dim grid sizes of 2^31, and so
    // will not exceed their grid size limit without also exceeding their memory
    // capacity.
#if __CUDA_ARCH__ < 300
    while (block < numBlocks)
#endif
    {
        int gid = NV * block;
        int count2 = min(NV, count - gid);

        int headFlags = mgpu::DeviceIndicesToHeadFlags<NT, VT>(indices_global,
            partitions_global, tid, block, count2, shared.words, shared.flags);

        mgpu::DeviceSegBlocksort<NT, VT, Stable, HasValues>(
            keys_global, values_global, count2, shared.keys, shared.values,
            shared.ranges, headFlags, tid, block, keysDest_global,
            valsDest_global, ranges_global, comp);

#if __CUDA_ARCH__ < 300
        __syncthreads();
        block += gridDim.x;
#endif
    }
}

// MGPU's SegSortPairsFromIndices in kernels/segmentedsort.cuh, using the above
// modified block sort kernel that tolerates numBlocks > device maximum.
// (All other kernels here are unchanged: they are similarly intolerant, but
// their grid sizes are small enough that we could not max out compute
// capability 2.0 hardware without first running out of memory.)
template<typename KeyType, typename ValType, typename Comp>
MGPU_HOST void SegSortPairsFromIndices(
    KeyType* keys_global,
    ValType* values_global,
    int count,
    const int* indices_global,
    int indicesCount,
    mgpu::CudaContext& context,
    Comp comp,
    bool verbose = false)
{
    const bool Stable = true;
    typedef mgpu::LaunchBoxVT<
        128, 11, 0,
        128, 7, 0,
        128, 7, 0
    > Tuning;
    int2 launch = Tuning::GetLaunchParams(context);
    const int NV = launch.x * launch.y;

    int numBlocks = MGPU_DIV_UP(count, NV);
    int deviceMaxBlocks = context.Device().Prop().maxGridSize[0];
    int numPasses = mgpu::FindLog2(numBlocks, true);

    mgpu::SegSortSupport support;
    MGPU_MEM(mgpu::byte) mem = mgpu::AllocSegSortBuffers(count, NV, support,
                                                         true, context);

    MGPU_MEM(KeyType) keysDestDevice = context.Malloc<KeyType>(count);
    MGPU_MEM(ValType) valsDestDevice = context.Malloc<ValType>(count);

    KeyType* keysSource = keys_global;
    KeyType* keysDest = keysDestDevice->get();
    ValType* valsSource = values_global;
    ValType* valsDest = valsDestDevice->get();

    MGPU_MEM(int) partitionsDevice =
        mgpu::BinarySearchPartitions<mgpu::MgpuBoundsLower>(
            count, indices_global, indicesCount, NV, mgpu::less<int>(),
            context);

    // Modified c.f. MGPU.
    grace::KernelSegBlocksortIndices<Tuning, Stable, true>
        <<<min(numBlocks, deviceMaxBlocks), launch.x, 0, context.Stream()>>>(
            keysSource, valsSource, count, numBlocks, indices_global,
            partitionsDevice->get(),
            (1 & numPasses) ? keysDest : keysSource,
            (1 & numPasses) ? valsDest : valsSource,
            support.ranges_global,
            comp);
    CUDA_HANDLE_ERR(cudaPeekAtLastError());
    MGPU_SYNC_CHECK("KernelSegBlocksortIndices");

    if(1 & numPasses) {
    std::swap(keysSource, keysDest);
    std::swap(valsSource, valsDest);
    }

    mgpu::SegSortPasses<Tuning, true, true>(support, keysSource, valsSource,
                                            count, numBlocks, numPasses,
                                            keysDest, valsDest, comp, context,
                                            verbose);
}

template <typename Float, typename T>
GRACE_HOST void sort_by_distance(
    thrust::device_vector<Float>& d_hit_distances,
    const thrust::device_vector<int>& d_ray_offsets,
    thrust::device_vector<unsigned int>& d_hit_indices,
    thrust::device_vector<T>& d_hit_data)
{
    // MGPU calls require a context.
    int device_ID = 0;
    cudaGetDevice(&device_ID);
    mgpu::ContextPtr mgpu_context_ptr = mgpu::CreateCudaDevice(device_ID);

    // d_sort_map will be used to reorder the input vectors.
    thrust::device_vector<unsigned int> d_sort_map(d_hit_distances.size());
    thrust::sequence(d_sort_map.begin(), d_sort_map.end(), 0u);

    // First, sort the hit distances and the indicies within the segments
    // defined by d_ray_offsets, i.e. sort each ray and its indices by distance.
    // The distances are the keys, and the ordered indices are the values.
    grace::SegSortPairsFromIndices(
        thrust::raw_pointer_cast(d_hit_distances.data()),
        thrust::raw_pointer_cast(d_sort_map.data()),
        d_hit_distances.size(),
        thrust::raw_pointer_cast(d_ray_offsets.data()),
        d_ray_offsets.size(),
        *mgpu_context_ptr,
        mgpu::less<Float>());

    // Second, reorder the hit indices and hit data by the map produced in the
    // above sort.
    order_by_index(d_sort_map, d_hit_indices);
    order_by_index(d_sort_map, d_hit_data);
}

} // namespace grace
