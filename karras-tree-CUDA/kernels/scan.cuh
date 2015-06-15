/*
 * Copyright (c) 2015 Sam Thomson
 *
 *  This file is free software: you may copy, redistribute and/or modify it
 *  under the terms of the GNU General Public License as published by the
 *  Free Software Foundation, either version 2 of the License, or (at your
 *  option) any later version.
 *
 *  This file is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *  General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * This file incorporates work covered by the following copyright and
 * permission notice:
 *
 *     Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
 *
 *     Redistribution and use in source and binary forms, with or without
 *     modification, are permitted provided that the following conditions are met:
 *         * Redistributions of source code must retain the above copyright
 *           notice, this list of conditions and the following disclaimer.
 *         * Redistributions in binary form must reproduce the above copyright
 *           notice, this list of conditions and the following disclaimer in the
 *           documentation and/or other materials provided with the distribution.
 *         * Neither the name of the NVIDIA CORPORATION nor the
 *           names of its contributors may be used to endorse or promote products
 *           derived from this software without specific prior written permission.
 *
 *     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *     ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 *     DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *     ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *     SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Original code and text by Sean Baxter, NVIDIA Research.
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 */

#pragma once

#include "../types.h"

#include "../../../moderngpu/include/mgpuhost.cuh"
#include "../../../moderngpu/include/mgpuenums.h"

#include "../../../moderngpu/include/device/ctascan.cuh"
#include "../../../moderngpu/include/device/ctasegreduce.cuh"
#include "../../../moderngpu/include/device/ctasegscan.cuh"
#include "../../../moderngpu/include/device/launchbox.cuh"
#include "../../../moderngpu/include/device/loadstore.cuh"

#include "../../../moderngpu/include/kernels/segreducecsr.cuh"

#include <assert.h>

namespace grace {

template<int NT_, int VT_, int OCC_, bool HalfCapacity_, bool LdgTranspose_>
struct SegScanTuning {
    enum {
        NT = NT_,
        VT = VT_,
        OCC = OCC_,
        HalfCapacity = HalfCapacity_,
        LdgTranspose = LdgTranspose_
    };
};

// SegScanCsrPreprocess and related.
// Optimial for segment sizes of ~1K+
template<size_t size>
struct SegScanPreprocessTuning {
    typedef mgpu::LaunchBox<
        // NOTE: due to implementation, max. values are (NT, VT) = (128, 13).
        grace::SegScanTuning<128, 7, 0, false, false>, // sm_20
        grace::SegScanTuning<128, 11, 0, true, false>, // sm_30
        grace::SegScanTuning<128, 11, 0, true, (size > 4) ? false : true> // sm_35
    > Tuning;
};

template<int NT, int VT, int Capacity, typename T, typename OutputIt>
MGPU_DEVICE void DeviceThreadToGlobalHalfCapacity(
    int count,
    const T* threadReg,
    int tid,
    OutputIt dest,
    T* shared,
    bool sync)
{
    // The below assumes that Capacity = NV/2 is a multiple of VT!
    assert(NT*VT % 2 == 0);

    // Write the first half of the block's NV/2 values.
    if ((tid + 1) * VT <= Capacity) {
        #pragma unroll
        for (int i = 0; i < VT; ++i)
            shared[VT * tid + i] = threadReg[i];
    }
    __syncthreads();

    int count2 = min(count, Capacity);
    mgpu::DeviceSharedToGlobal<NT, VT>(count2, shared, tid, dest, true);

    // Write the second half.
    if (tid * VT >= Capacity) {
        #pragma unroll
        for (int i = 0; i < VT; ++i)
            shared[(VT * tid) + i - Capacity] = threadReg[i];
    }
    __syncthreads();

    count2 = min(count - Capacity, Capacity);
    if (count2 > 0) {
        dest += Capacity;
        mgpu::DeviceSharedToGlobal<NT, VT>(count2, shared, tid, dest, sync);
    }
}

// Copy of MGPU's KernelSegReduceApply in kernels/segreducecsr.cuh, modified to
// output the scan result rather than the reduction. Calls some similarly
// modified device functions where necessary.
template<typename Tuning, typename InputIt, typename DestIt, typename T,
         typename Op>
MGPU_LAUNCH_BOUNDS void KernelSegScanApply(
    const int* threadCodes_global,
    int count,
    int numBlocks,
    const int* limits_global,
    InputIt data_global,
    T identity,
    Op op,
    DestIt dest_global,
    T* carryOut_global)

{
    typedef MGPU_LAUNCH_PARAMS Params;
    const int NT = Params::NT;
    const int VT = Params::VT;
    const int NV = NT * VT;
    const bool HalfCapacity =
        (sizeof(T) > sizeof(int)) && Params::HalfCapacity;
    const bool LdgTranspose = Params::LdgTranspose;

    typedef mgpu::CTAScan<NT, Op> FastScan;
    typedef mgpu::CTASegScan<NT, Op> SegScan;
    typedef mgpu::CTASegReduceLoad<NT, VT, HalfCapacity, LdgTranspose, T>
        SegReduceLoad;

    enum { Capacity = HalfCapacity ? (NV / 2) : NV };

    union Shared {
        int csr[NV];
        T values[Capacity];
        typename FastScan::Storage scanStorage;
        typename SegScan::Storage segScanStorage;
        typename SegReduceLoad::Storage loadStorage;
    };
    __shared__ Shared shared;

    int tid = threadIdx.x;
    int block = blockIdx.x;

#if __CUDA_ARCH__ < 300
    while (block < numBlocks)
#endif
    {
        int gid = NV * block;
        int count2 = min(NV, count - gid);

        int limit0 = limits_global[block];
        int limit1 = limits_global[block + 1];
        int threadCodes = threadCodes_global[NT * block + tid];

        // Load the data and transpose into thread order.
        T data[VT];
        SegReduceLoad::LoadDirect(count2, tid, gid, data_global, identity,
                                  data, shared.loadStorage);

        // Compute the range.
        mgpu::SegReduceRange range = mgpu::DeviceShiftRange(limit0, limit1);

        T x = identity;
        T localScan[VT];
        T carryOut;
        if(range.total) {
            // This block contains end flags, so we must take care to only
            // propagate carry-ins (x) to as many values as required.
            int segs[VT + 1];
            mgpu::DeviceExpandFlagsToRows<VT>(threadCodes >> 20, threadCodes,
                                              segs);

            // Intra-thread exclusive scan.
            #pragma unroll
            for(int i = 0; i < VT; ++i) {
                localScan[i] = x;
                x = i ? op(x, data[i]) : data[i];
                // Clear the carry-in on an end flag.
                if(segs[i] != segs[i + 1])
                    x = identity;
            }

            // Scan sum leftward-threads' carry-outs.
            int tidDelta = 0x7f & (threadCodes>> 13);
            x = SegScan::SegScanDelta(tid, tidDelta, x, shared.segScanStorage,
                                      &carryOut, identity, op);

            // Add carry-in to each thread-local scan value if required.
            #pragma unroll
            for(int i = 0; i < VT; ++i) {
                localScan[i] = op(x, localScan[i]);
                // Clear the carry-in on an end flag.
                if(segs[i] != segs[i + 1])
                    x = identity;
            }
        }
        else {
            // There are no end flags in this CTA, so we use a fast scan (a
            // normal CTAScan) and propagate carry-ins (x) to all elements in
            // the thread.

            // Intra-thread exclusive scan.
            #pragma unroll
            for(int i = 0; i < VT; ++i) {
                localScan[i] = x;
                x = i ? op(x, data[i]) : data[i];
            }

            // Scan sum leftward-threads' carry-outs.
            x = FastScan::Scan(tid, x, shared.scanStorage, &carryOut,
                               mgpu::MgpuScanTypeExc, identity, op);

            // Add carry-in to each thread-local scan value.
            #pragma unroll
            for(int i = 0; i < VT; ++i)
                localScan[i] = op(x, localScan[i]);
        }

        if(!tid)
            carryOut_global[block] = carryOut;

        // Cooperatively store partial scans to global memory.
        DestIt dest_block = dest_global + (NV * block);
        if (HalfCapacity) {
            // Not all partial scans fit in shared memory.
            grace::DeviceThreadToGlobalHalfCapacity<NT, VT, Capacity>(
                count2, localScan, tid, dest_block, shared.values, false);
        }
        else {
            mgpu::DeviceThreadToShared<VT>(localScan, tid, shared.values,
                                           true);
            mgpu::DeviceSharedToGlobal<NT, VT>(count2, shared.values, tid,
                                               dest_block, false);
        }

#if __CUDA_ARCH__ < 300
        __syncthreads();
        block += gridDim.x;
#endif
    }
}

///////////////////////////////////////////////////////////////////////////////
// SegScanSpine
// Compute the carry-in in-place. Return the carry-out for the entire tile.
// A final spine-reducer scans the tile carry-outs and adds into individual
// results.

// Copy of MGPU's KernelSegReduceSpine1 in kernels/segreduce.cuh, but modified
// for scanning rather than reduction.
template<int NT, typename T, typename Op>
__global__ void KernelSegScanSpine1(
    const int* limits_global,
    int count,
    T* carryIn_global,
    T identity,
    Op op,
    T* carryOut_global)
{
    typedef mgpu::CTASegScan<NT, Op> SegScan;
    union Shared {
        typename SegScan::Storage segScanStorage;
    };
    __shared__ Shared shared;

    int tid = threadIdx.x;
    int block = blockIdx.x;
    int gid = NT * block + tid;

    // Load the current carry-in and the current and next row indices.
    int row = (gid < count) ?
        (0x7fffffff & limits_global[gid]) :
        INT_MAX;
    int row2 = (gid + 1 < count) ?
        (0x7fffffff & limits_global[gid + 1]) :
        INT_MAX;

    T carryIn2 = (gid < count) ? carryIn_global[gid] : identity;

    // Run a segmented scan of the carry-in values.
    bool endFlag = row != row2;

    T carryOut;
    T x = SegScan::SegScan(tid, carryIn2, endFlag, shared.segScanStorage,
                           &carryOut, identity, op);

    // Update the carry-in of block gid in KernelSegScanApply.  This now
    // includes relevant carry-outs from blocks [NT * block, gid).
    if (gid < count)
        carryIn_global[gid] = x;

    // Store the CTA carry-out.  This is the carry-out of the group of blocks
    // [NT * block, NT * (block + 1)) in KernelSegScanApply.
    if(!tid) carryOut_global[block] = carryOut;
}

// Copy of MGPU's KernelSegReduceSpine2 in kernels/segreduce.cuh, modified to
// output the scan result rather than reduction.
template<int NT, typename T, typename Op>
__global__ void KernelSegScanSpine2a(
    const int* limits_global,
    int numBlocks,
    int count,
    int nv,
    T* carryIn_global,
    T identity,
    Op op)
{
    typedef mgpu::CTASegScan<NT, Op> SegScan;
    struct Shared {
        typename SegScan::Storage segScanStorage;
        int carryInRow;
        T carryIn;
    };
    __shared__ Shared shared;

    int tid = threadIdx.x;

    for(int i = 0; i < numBlocks; i += NT) {
        int gid = (i + tid) * nv;

        // Load the current carry-in and the current and next row indices.
        int row = (gid < count) ?
            (0x7fffffff & limits_global[gid]) : INT_MAX;
        int row2 = (gid + nv < count) ?
            (0x7fffffff & limits_global[gid + nv]) : INT_MAX;
        T carryIn2 = (i + tid < numBlocks) ?
            carryIn_global[i + tid] : identity;

        // Run a segmented scan of the carry-in values.
        bool endFlag = row != row2;

        T carryOut;
        T x = SegScan::SegScan(tid, carryIn2, endFlag, shared.segScanStorage,
                               &carryOut, identity, op);

        // Update the carry-in of block i + tid in KernelSegScanSpine1.
        // This now includes relevant carry-outs all prior blocks.
        if(i && row == shared.carryInRow) {
            // Add the carry-in from the last loop iteration to the carry-in
            // from this loop iteration.
            x = op(shared.carryIn, x);
        }
        if (i + tid < numBlocks)
            carryIn_global[i + tid] = x;

        // Set the carry-in for the next loop iteration.
        if(i + NT < numBlocks) {
            __syncthreads();
            if(i > 0) {
                // Add in the previous carry-in.
                if(NT - 1 == tid) {
                    shared.carryIn = (shared.carryInRow == row2) ?
                        op(shared.carryIn, carryOut) : carryOut;
                    shared.carryInRow = row2;
                }
            } else {
                if(NT - 1 == tid) {
                    shared.carryIn = carryOut;
                    shared.carryInRow = row2;
                }
            }
            __syncthreads();
        }
    }
}

template<int NT, typename T, typename Op>
__global__ void KernelSegScanSpine2b(
    const int* limits_global,
    int count,
    const T* carryIn2_global,
    T* carryIn_global,
    T identity,
    Op op)
{
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int gid = NT * block + tid;

    if (gid < count)
    {
        // Load the current row index and the row index of the first thread in
        // this block --- the carryIn2 values only apply to the segment that
        // spans the first thread in this block.
        int firstRow = (0x7fffffff & limits_global[NT * block]);
        int row = (0x7fffffff & limits_global[gid]);

        if (row == firstRow) {
            T carryIn = carryIn_global[gid];
            T carryIn2 = carryIn2_global[block];

            // Update the carry-in of block gid in KernelSegScanApply.  This
            // now includes relevant carry-outs from all prior blocks.
            carryIn_global[gid] = op(carryIn2, carryIn);
        }
    }
}

template<typename Tuning, typename ScanIt, typename T, typename Op>
MGPU_LAUNCH_BOUNDS void KernelSegScanSpineApply(
    const int* threadCodes_global,
    const int* limits_global,
    int count,
    int numBlocks,
    const T* carryIn_global,
    ScanIt scan_global,
    T identity,
    Op op)
{
    typedef MGPU_LAUNCH_PARAMS Params;
    const int NT = Params::NT;
    const int VT = Params::VT;
    const int NV = NT * VT;
    const bool HalfCapacity =
        (sizeof(T) > sizeof(int)) && Params::HalfCapacity;
    const bool LdgTranspose = Params::LdgTranspose;

    typedef mgpu::CTASegReduceLoad<NT, VT, HalfCapacity, LdgTranspose, T>
        SegReduceLoad;

    enum { Capacity = HalfCapacity ? (NV / 2) : NV };

    union Shared {
        T values[Capacity];
        typename SegReduceLoad::Storage loadStorage;
    };
    __shared__ Shared shared;
    // firstSeg stores the segment to which the first element in the first
    // thread belongs.
    __shared__ int firstSeg;
    // inFirstSeg determines which threads have elements in the first segment.
    __shared__ unsigned int inFirstSeg[(NT + 31)/32];

    int tid = threadIdx.x;
    int block = blockIdx.x;

# if __CUDA_ARCH__ < 300
    while (block < numBlocks)
#endif
    {
        int gid = NV * block;
        int count2 = min(NV, count - gid);

        int limit0 = limits_global[block];
        int limit1 = limits_global[block + 1];
        int threadCodes = threadCodes_global[NT * block + tid];

        T carryIn = carryIn_global[block];

        // Compute the range.
        mgpu::SegReduceRange range = mgpu::DeviceShiftRange(limit0, limit1);

        // Update all scan values with their respective carry-ins.
        if (range.total)
        {
            // Load the data (scan sum without all carry-ins applied) and
            // transpose into thread order.
            T data[VT];
            SegReduceLoad::LoadDirect(count2, tid, gid, scan_global, identity,
                                      data, shared.loadStorage);

            // Expand the segment indices.
            int segs[VT + 1];
            mgpu::DeviceExpandFlagsToRows<VT>(threadCodes >> 20, threadCodes,
                                              segs);

            if (tid == 0)
                firstSeg = segs[0];
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < VT; ++i) {
                if (segs[i] == firstSeg) {
                    data[i] = op(carryIn, data[i]);
                }
            }

            // Quickly obtain an upper bound (nearest multiple of VT) for the
            // number of elements in this block which have been updated and
            // need to be stored back to global.
            int outCount = 0;
            unsigned int bitfield = __ballot(segs[0] == firstSeg);
            int lane = (31 & tid);
            int wid = tid/32;
            if (lane == 0)
                inFirstSeg[wid] = bitfield;
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < (NT + 31)/32; ++i) {
                bitfield = inFirstSeg[i];
                if (bitfield == uint(-1))
                    outCount += 32 * VT;
                else {
                    // The first zero bit identifies the first thread which
                    // contains no updated elements, and hence the upper bound.
                    outCount += (32 - __clz(bitfield)) * VT;
                    break;
                }
            }

            // Cooperatively store scans back to global memory.
            ScanIt scan_block = scan_global + (NV * block);
            if (HalfCapacity) {
                grace::DeviceThreadToGlobalHalfCapacity<NT, VT, Capacity>(
                    outCount, data, tid, scan_block, shared.values, false);
            }
            else {
                mgpu::DeviceThreadToShared<VT>(data, tid, shared.values, true);
                mgpu::DeviceSharedToGlobal<NT, VT>(
                    outCount, shared.values, tid, scan_block, false);
            }
        }
        else {
            // This block has no end flags, so we can apply the block's carry-in to
            // all elements within it.
            T data[VT];
            ScanIt scan_block = scan_global + (NV * block);

            mgpu::DeviceGlobalToReg<NT, VT>(count2, scan_block, tid, data,
                                            false);

            #pragma unroll
            for (int i = 0; i < VT; ++i)
                data[i] = op(carryIn, data[i]);

            mgpu::DeviceRegToGlobal<NT, VT>(count2, data, tid, scan_block,
                                            false);
        }

#if __CUDA_ARCH__ < 300
        __syncthreads();
        block += gridDim.x;
#endif
    }
}

// Copy of MGPU's SegReduceSpine in kernels/segreduce.cuh, but calling modified
// kernels for scanning rather than reduction.
template<typename T, typename Op, typename InputIt, typename DestIt>
MGPU_HOST void SegScanSpine(
    const int* threadCodes_global,
    const int* limits_global,
    const InputIt data_global,
    int count,
    int numBlocks,
    int numLaunchBlocks,
    DestIt dest_global,
    T* carryIn_global,
    T identity,
    Op op,
    mgpu::CudaContext& context)
{
    const int NT = 128;
    int numBlocks2 = MGPU_DIV_UP(numBlocks, NT);

    // Fix-up the carry-ins between the original tiles.
    MGPU_MEM(T) carryOutDevice = context.Malloc<T>(numBlocks2);
    KernelSegScanSpine1<NT><<<numBlocks2, NT, 0, context.Stream()>>>(
        limits_global, numBlocks, carryIn_global, identity, op,
        carryOutDevice->get());
    MGPU_SYNC_CHECK("KernelSegScanSpine1");

    // Fix-up the carry-ins between the tiles of KernelSegScanSpine1.
    if(numBlocks2 > 1) {
        // Compute and save all carry-ins for the tiles of KenelSegScanSpine1
        KernelSegScanSpine2a<NT><<<1, NT, 0, context.Stream()>>>(
            limits_global, numBlocks2, numBlocks, NT, carryOutDevice->get(),
            identity, op);
        MGPU_SYNC_CHECK("KernelSegScanSpine2");

        // Update all carry-ins for the original tiles.
        KernelSegScanSpine2b<NT><<<numBlocks2, NT, 0, context.Stream()>>>(
            limits_global, numBlocks, carryOutDevice->get(), carryIn_global,
            identity, op);
        MGPU_SYNC_CHECK("KernelSegScanSpine2b");
    }

    typedef typename grace::SegScanPreprocessTuning<sizeof(T)>::Tuning Tuning;
    int2 launch = Tuning::GetLaunchParams(context);
    KernelSegScanSpineApply<Tuning>
        <<<numLaunchBlocks, launch.x, 0, context.Stream()>>>(
            threadCodes_global, limits_global, count, numBlocks,
            carryIn_global, dest_global, identity, op);
    MGPU_SYNC_CHECK("KernelSegScanSpineApply");
}

///////////////////////////////////////////////////////////////////////////////
// BuildCsrPlus

// Copy of MGPU's (Kernel)BuildCsrPlus, but modified to be tolerant of a
// numBlocks value exceeding the device limit.

template<typename Tuning, typename CsrIt>
MGPU_LAUNCH_BOUNDS void KernelBuildCsrPlus(
    int count,
    int numBlocks,
    CsrIt csr_global,
    const int* limits_global,
    int* threadCodes_global) {

    typedef MGPU_LAUNCH_PARAMS Params;
    const int NT = Params::NT;
    const int VT = Params::VT;
    const int NV = NT * VT;

    union Shared {
        int csr[NV + 1];
        typename mgpu::CTAScan<NT>::Storage scanStorage;
    };
    __shared__ Shared shared;

    int tid = threadIdx.x;
    int block = blockIdx.x;

#if __CUDA_ARCH__ < 300
    while (block < numBlocks)
#endif
    {
        int gid = NV * block;
        int count2 = min(NV, count - gid);

        int limit0 = limits_global[block];
        int limit1 = limits_global[block + 1];

        // Transform the row limits into ranges.
        mgpu::SegReduceRange range = mgpu::DeviceShiftRange(limit0, limit1);
        int numRows = range.end - range.begin;

        // Load the Csr interval.
        mgpu::DeviceGlobalToSharedLoop<NT, VT>(
            numRows, csr_global + range.begin, tid, shared.csr);

        // Flatten Csr->COO and return the segmented scan terms.
        int rows[VT + 1], rowStarts[VT];
        mgpu::SegReduceTerms terms = mgpu::DeviceSegReducePrepare<NT, VT>(
            shared.csr, numRows, tid, gid, range.flushLast, rows, rowStarts);

        // Combine terms into bit field.
        // threadCodes:
        // 12:0 - end flags for up to 13 values per thread.
        // 19:13 - tid delta for up to 128 threads.
        // 30:20 - scan offset for streaming partials.
        int threadCodes = terms.endFlags
                          | (terms.tidDelta<< 13)
                          | (rows[0]<< 20);
        threadCodes_global[NT * block + tid] = threadCodes;

#if __CUDA_ARCH__ < 300
        __syncthreads();
        block += gridDim.x;
#endif
    }
}

template<typename Tuning, typename CsrIt>
MGPU_HOST MGPU_MEM(int) BuildCsrPlus(
    int count,
    CsrIt csr_global,
    const int* limits_global,
    int numBlocks,
    int numLaunchBlocks,
    mgpu::CudaContext& context) {

    int2 launch = Tuning::GetLaunchParams(context);

    // Allocate one int per thread.
    MGPU_MEM(int) threadCodesDevice =
        context.Malloc<int>(launch.x * numBlocks);

    KernelBuildCsrPlus<Tuning>
        <<<numLaunchBlocks, launch.x, 0, context.Stream()>>>(
            count, numBlocks, csr_global, limits_global,
            threadCodesDevice->get());
    MGPU_SYNC_CHECK("KernelBuildCsrPlus");

    return threadCodesDevice;
}

///////////////////////////////////////////////////////////////////////////////
// SegScanPreprocess

// Copy of MGPU's SegReducePreprocessData in kernels/segreduce.cuh, with the
// additional numLaunchBlocks field, in the case that numBlocks exceeds the
// device's capabilities.

struct SegScanPreprocessData {
    int count, numSegments, numSegments2;
    int numBlocks, numLaunchBlocks;
    MGPU_MEM(int) limitsDevice;
    MGPU_MEM(int) threadCodesDevice;

    // If csr2Device is set, use BulkInsert to finalize results into
    // dest_global.
    MGPU_MEM(int) csr2Device;
};

// Copy of MGPU's SegScanPreprocess in kernels/segreduce.cuh, but calling a
// modified BuildCsrPlus to handle numBlocks > device limit.
template<typename Tuning, typename CsrIt>
MGPU_HOST void SegScanPreprocess(
    int count,
    CsrIt csr_global,
    int numSegments,
    bool supportEmpty,
    std::auto_ptr<SegScanPreprocessData>* ppData,
    mgpu::CudaContext& context) {

    std::auto_ptr<SegScanPreprocessData> data(new SegScanPreprocessData);

    int2 launch = Tuning::GetLaunchParams(context);
    int NV = launch.x * launch.y;

    int numBlocks = MGPU_DIV_UP(count, NV);
    int deviceMaxBlocks = context.Device().Prop().maxGridSize[0];
    int numLaunchBlocks = min(numBlocks, deviceMaxBlocks);
    data->count = count;
    data->numSegments = data->numSegments2 = numSegments;
    data->numBlocks = numBlocks;
    data->numLaunchBlocks = numLaunchBlocks;

    // Filter out empty rows and build a replacement structure.
    if(supportEmpty) {
        MGPU_MEM(int) csr2Device = context.Malloc<int>(numSegments + 1);
        mgpu::CsrStripEmpties<false>(
            count, csr_global, (const int*)0, numSegments, csr2Device->get(),
            (int*)0, (int*)&data->numSegments2, context);
        if(data->numSegments2 < numSegments) {
            csr_global = csr2Device->get();
            numSegments = data->numSegments2;
            data->csr2Device = csr2Device;
        }
    }

    data->limitsDevice = mgpu::PartitionCsrSegReduce(count, NV, csr_global,
        numSegments, (const int*)0, numBlocks + 1, context);
    data->threadCodesDevice = BuildCsrPlus<Tuning>(count, csr_global,
        data->limitsDevice->get(), numBlocks, numLaunchBlocks, context);

    *ppData = data;
}

template<typename T, typename CsrIt>
MGPU_HOST void SegScanCsrPreprocess(
    int count, CsrIt csr_global,
    int numSegments,
    bool supportEmpty,
    std::auto_ptr<SegScanPreprocessData>* ppData,
    mgpu::CudaContext& context)
{
    // The scan preprocessing required is identical to SegReduceCsr.
    // This first allocates an array of size (numSegments + 1), if supportEmpty
    // is true, and fills it with segment offsets, filtering out the
    // zero-length segments.
    // Then each block's segment limits and per-thread data required for the
    // scans are computed and saved.
    typedef typename grace::SegScanPreprocessTuning<sizeof(T)>::Tuning Tuning;
    SegScanPreprocess<Tuning>(count, csr_global, numSegments, supportEmpty,
                              ppData, context);
}

// Copy of MGPU's SegReduceApply in kernels/segreducecsr.cuh, but calling
// modified kernels for scanning rather than reduction.
template<typename InputIt, typename DestIt, typename T, typename Op>
MGPU_HOST void SegScanApply(
    const SegScanPreprocessData& preprocess,
    InputIt data_global,
    T identity,
    Op op,
    DestIt dest_global,
    mgpu::CudaContext& context)
{
    typedef typename grace::SegScanPreprocessTuning<sizeof(T)>::Tuning Tuning;
    int2 launch = Tuning::GetLaunchParams(context);

    // Set the bank-size to eight bytes if we are dealing with > 4-byte types.
    // Also save the current state so it can be reset later.
    // On a GTX 670, this seems to give slightly worse results for T = double.
    // cudaSharedMemConfig pConfig;
    // cudaDeviceGetSharedMemConfig(&pConfig);
    // if (sizeof(T) > sizeof(int))
    //     cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    MGPU_MEM(T) carryOutDevice = context.Malloc<T>(preprocess.numBlocks);
    KernelSegScanApply<Tuning>
        <<<preprocess.numLaunchBlocks, launch.x, 0, context.Stream()>>>(
            preprocess.threadCodesDevice->get(), preprocess.count,
            preprocess.numBlocks, preprocess.limitsDevice->get(), data_global,
            identity, op, dest_global, carryOutDevice->get());
    // TODO: Replace this with own version.
    MGPU_SYNC_CHECK("KernelSegScanApply");

    // Scan the per-block block carry-ins to propagate all carry-ins
    // across the entire the output (e.g. with only one segment, the final
    // block's carry-in is the exclusive scan of all prior blocks'
    // carry-outs.
    SegScanSpine(preprocess.threadCodesDevice->get(),
                 preprocess.limitsDevice->get(), data_global,
                 preprocess.count, preprocess.numBlocks,
                 preprocess.numLaunchBlocks, dest_global,
                 carryOutDevice->get(), identity, op, context);

    // Reset the bank-size to its original state.
    // cudaDeviceSetSharedMemConfig(pConfig);
}

template <typename Real>
GRACE_HOST void exclusive_segmented_scan(
    const thrust::device_vector<int>& d_segment_offsets,
    thrust::device_vector<Real>& d_data,
    thrust::device_vector<Real>& d_results)
{
    // MGPU calls require a context.
    int device_ID = 0;
    cudaGetDevice(&device_ID);
    mgpu::ContextPtr mgpu_context_ptr = mgpu::CreateCudaDevice(device_ID);

    std::auto_ptr<grace::SegScanPreprocessData> pp_data_ptr;

    // TODO: Can the segmented scan be done in-place?
    size_t N_data = d_data.size();
    size_t N_segments = d_segment_offsets.size();

    SegScanCsrPreprocess<Real>(
        N_data, thrust::raw_pointer_cast(d_segment_offsets.data()),
        N_segments, true, &pp_data_ptr, *mgpu_context_ptr);

    SegScanApply(*pp_data_ptr, thrust::raw_pointer_cast(d_data.data()),
                 Real(0), mgpu::plus<Real>(),
                 thrust::raw_pointer_cast(d_results.data()), *mgpu_context_ptr);
}

} // namespace grace
