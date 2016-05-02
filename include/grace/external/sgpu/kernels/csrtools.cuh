/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION; 2016, Sam Thomson.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/******************************************************************************
 *
 * Original code and text by Sean Baxter, NVIDIA Research
 * Modified code and text by Sam Thomson.
 * Segmented GPU is a derivative of Modern GPU.
 * See http://nvlabs.github.io/moderngpu for original repository and
 * documentation.
 *
 ******************************************************************************/

#pragma once

#include "../device/ctasegreduce.cuh"
#include "../kernels/scan.cuh"
#include "../kernels/bulkinsert.cuh"

namespace sgpu {

////////////////////////////////////////////////////////////////////////////////
// KernelPartitionCsrPlus
// Standard upper-bound partitioning.

// Not tolerant to large block sizes, but do not expect them.
template<int NT, typename CsrIt>
__global__ void KernelPartitionCsrPlus(int nz, int nv, CsrIt csr_global,
	int numRows, const int* numRows2, int numPartitions, int* limits_global) {

	if(numRows2) numRows = *numRows2;

	int gid = NT * blockIdx.x + threadIdx.x;
	if(gid < numPartitions) {
		int key = min(nv * gid, nz);

		int ub;
		if(key == nz) ub = numRows;
		else {
			// Upper-bound search for this partition.
			ub = BinarySearch<SgpuBoundsUpper>(csr_global, numRows, key,
				sgpu::less<int>()) - 1;

			// Check if limit points to matching value.
			if(key != csr_global[ub]) ub |= 0x80000000;
		}
		limits_global[gid] = ub;
	}
}

template<typename CsrIt>
SGPU_HOST SGPU_MEM(int) PartitionCsrPlus(int count, int nv,
	CsrIt csr_global, int numRows, const int* numRows2, int numPartitions,
	CudaContext& context) {

	// Allocate one int per partition.
	SGPU_MEM(int) limitsDevice = context.Malloc<int>(numPartitions);

	int numBlocks2 = SGPU_DIV_UP(numPartitions, 64);
	KernelPartitionCsrPlus<64><<<numBlocks2, 64, 0, context.Stream()>>>(
		count, nv, csr_global, numRows, numRows2, numPartitions,
		limitsDevice->get());
	SGPU_SYNC_CHECK("KernelPartitionCsrPlus");

	return limitsDevice;
}

////////////////////////////////////////////////////////////////////////////////
// KernelBuildCsrPlus

template<typename Tuning, typename CsrIt>
SGPU_LAUNCH_BOUNDS void KernelBuildCsrPlus(int numTiles, int count,
	CsrIt csr_global, const int* limits_global, int* threadCodes_global) {

	typedef SGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;

	union Shared {
		int csr[NV + 1];
		typename CTAScan<NT>::Storage scanStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;

	for(int block = blockIdx.x; block < numTiles; block += gridDim.x) {
		__syncthreads();

		int gid = NV * block;
		int count2 = min(NV, count - gid);

		int limit0 = limits_global[block];
		int limit1 = limits_global[block + 1];

		// Transform the row limits into ranges.
		SegReduceRange range = DeviceShiftRange(limit0, limit1);
		int numRows = range.end - range.begin;

		// Load the Csr interval.
		DeviceGlobalToSharedLoop<NT, VT>(numRows, csr_global + range.begin, tid,
			shared.csr);

		// Flatten Csr->COO and return the segmented scan terms.
		int rows[VT + 1], rowStarts[VT];
		SegReduceTerms terms = DeviceSegReducePrepare<NT, VT>(shared.csr,
			numRows, tid, gid, range.flushLast, rows, rowStarts);

		// Combine terms into bit field.
		// threadCodes:
		// 12:0 - end flags for up to 13 values per thread.
		// 19:13 - tid delta for up to 128 threads.
		// 30:20 - scan offset for streaming partials.
		int threadCodes = terms.endFlags | (terms.tidDelta<< 13) | (rows[0]<< 20);
		threadCodes_global[NT * block + tid] = threadCodes;
	}
}

template<typename Tuning, typename CsrIt>
SGPU_HOST SGPU_MEM(int) BuildCsrPlus(int count, CsrIt csr_global,
	const int* limits_global, int numTiles, CudaContext& context) {

	int2 launch = Tuning::GetLaunchParams(context);
	int maxBlocks = context.MaxGridSize();
	int numBlocks = min(numTiles, maxBlocks);

	// Allocate one int per thread.
	SGPU_MEM(int) threadCodesDevice = context.Malloc<int>(launch.x * numTiles);

	KernelBuildCsrPlus<Tuning>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(numTiles, count,
		csr_global, limits_global, threadCodesDevice->get());
	SGPU_SYNC_CHECK("KernelBuildCsrPlus");

	return threadCodesDevice;
}

////////////////////////////////////////////////////////////////////////////////
// CsrStripEmpties
// Removes empty rows from a Csr array. The returned array has numRows2
// non-empty rows in the front followed by (numRows - numRows2) BulkInsert
// offsets in the back.

template<typename Tuning, typename CsrIt>
SGPU_LAUNCH_BOUNDS void KernelCsrStripEmptiesUpsweep(int numTiles, int nz,
	CsrIt csr_global, int numRows, int* counts_global) {

	typedef SGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;

	union Shared {
		int indices[NT * (VT + 1)];
		typename CTAReduce<NT, int>::Storage reduceStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;

	for(int block = blockIdx.x; block < numTiles; block += gridDim.x) {
		__syncthreads();

		int gid = NV * block;
		int numRows2 = min(NV + 1, numRows - gid);

		// Load one index per element plus one halo element.
		DeviceGlobalToSharedDefault2<NT, VT, VT + 1>(numRows2, csr_global + gid,
			tid, shared.indices, nz);

		// Count the number of valid rows.
		int validRowCount = 0;
		int row = shared.indices[VT * tid];
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int next = shared.indices[VT * tid + 1 + i];
			validRowCount += row != next;
			row = next;
		}
		__syncthreads();

		validRowCount = CTAReduce<NT, int>::Reduce(tid, validRowCount,
			shared.reduceStorage);
		if(!tid)
			counts_global[block] = validRowCount;
	}
}

template<typename Tuning, bool Indirect, typename CsrIt, typename SourcesIt>
SGPU_LAUNCH_BOUNDS void KernelCsrStripEmptiesDownsweep(int numTiles, int nz,
	CsrIt csr_global, SourcesIt sources_global, int numRows,
	const int* scan_global,	int* csr2_global, int* sources2_global) {

	typedef SGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;

	union Shared {
		int indices[NT * (VT + 1)];
		typename CTAScan<NT>::Storage scanStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;

	for(int block = blockIdx.x; block < numTiles; block += gridDim.x) {
		__syncthreads();

		int gid = NV * block;
		int numRows2 = min(NV + 1, numRows - gid);

		// The total number of valid rows is at the end of the scan array.
		int invalidStart = scan_global[gridDim.x];

		// scan_global is the offset at which to store the first valid row offset.
		int scanOffset = scan_global[block];

		// Load one index per element plus one halo element.
		DeviceGlobalToSharedDefault2<NT, VT, VT + 1>(numRows2, csr_global + gid,
			tid, shared.indices, nz);

		// Count the number of skipped (empty) rows.
		int validRowCount = 0;
		int rows[VT + 1];
		rows[0] = shared.indices[VT * tid];
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			rows[i + 1] = shared.indices[VT * tid + i + 1];
			validRowCount += rows[i] != rows[i + 1];
		}
		__syncthreads();

		// Scan the number valid rows.
		int totalValid;
		int scan = CTAScan<NT>::Scan(tid, validRowCount, shared.scanStorage,
			&totalValid);
		int validScan = scan;
		int invalidScan = totalValid + VT * tid - validScan;

		// Stream the valid row offsets to the front and the invalid row indices
		// to the back.
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = VT * tid + i;
			bool invalid = index >= numRows2 || rows[i] == rows[i + 1];
			int dest = invalid ? invalidScan++ : validScan++;
			shared.indices[dest] = invalid ? (gid + index) : rows[i];
		}
		__syncthreads();

		// Cooperatively store valid row offsets and invalid row insertion points
		// to global memory.
		for(int i = tid; i < totalValid; i += NT)
			csr2_global[scanOffset + i] = shared.indices[i];

		numRows2 = min(NV, numRows2);
		int invalidRank = gid - scanOffset;
		int totalInvalid = numRows2 - totalValid;
		for(int i = tid; i < totalInvalid; i += NT) {
			// Subtract the rank of the invalid row from its index. This serves as
			// a BulkInsert-style insertion point.
			int invalidRank2 = invalidRank + i;
			csr2_global[invalidStart + invalidRank2] =
				shared.indices[totalValid + i] - invalidRank2;
		}
		__syncthreads();

		// Store the total number of valid rows to the end of the Csr2 array.
		if(!block && !tid)
			csr2_global[numRows] = invalidStart;

		if(Indirect) {
			// Load the indirect source offsets in thread order.
			int sources[VT];
			DeviceGlobalToShared<NT, VT>(numRows2, sources_global + gid, tid,
				shared.indices);
			DeviceSharedToThread<VT>(shared.indices, tid, sources);

			// Compact the valid source offsets to smem.
			#pragma unroll
			for(int i = 0; i < VT; ++i)
				if(rows[i] != rows[i + 1])
					shared.indices[scan++] = sources[i];
			__syncthreads();

			// Cooperatively store compacted source offsets.
			for(int i = tid; i < totalValid; i += NT)
				sources2_global[scanOffset + i] = shared.indices[i];
		}
	}
}

template<bool Indirect, typename CsrIt, typename SourcesIt>
SGPU_HOST void CsrStripEmpties(int nz, CsrIt csr_global,
	SourcesIt sources_global, int numRows, int* csr2_global,
	int* sources2_global, int* numRows2, CudaContext& context) {

	typedef LaunchBoxVT<
		128, 3, 0,
		128, 3, 0,
		128, 3, 0
	> Tuning;
	int2 launch = Tuning::GetLaunchParams(context);
	int NV = launch.x * launch.y;

	int numTiles = SGPU_DIV_UP(numRows, NV);
	SGPU_MEM(int) countsDevice = context.Malloc<int>(numTiles + 1);

	int maxBlocks = context.MaxGridSize();
	int numBlocks = min(numTiles, maxBlocks);

	// Count the non-empty row counts for each CTA.
	KernelCsrStripEmptiesUpsweep<Tuning>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(
		numTiles, nz, csr_global, numRows, countsDevice->get());
	SGPU_SYNC_CHECK("KernelCsrStripEmpties");

	// Scan the non-empty row counts.
	Scan<SgpuScanTypeExc>(countsDevice->get(), numTiles, 0, sgpu::plus<int>(),
		countsDevice->get() + numTiles, numRows2, countsDevice->get(),
		context);

	// Compact the non-empty rows to the front. Append the indices of all
	// empty rows to the end of the array.
	KernelCsrStripEmptiesDownsweep<Tuning, Indirect>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(
		numTiles, nz, csr_global, sources_global, numRows, countsDevice->get(),
		csr2_global, sources2_global);
	SGPU_SYNC_CHECK("KernelCsrStringEmptiesDownsweep");
}

////////////////////////////////////////////////////////////////////////////////
// SegCsrPreprocess

struct SegCsrPreprocessData {
	int count, numSegments, numSegments2;
	int numTiles;
	SGPU_MEM(int) limitsDevice;
	SGPU_MEM(int) threadCodesDevice;

	// If csr2Device is set, use BulkInsert to finalize results into
	// dest_global.
	SGPU_MEM(int) csr2Device;
};

// Generic function for prep
template<typename Tuning, typename CsrIt>
SGPU_HOST void SegCsrPreprocess(int count, CsrIt csr_global, int numSegments,
	bool supportEmpty, std::auto_ptr<SegCsrPreprocessData>* ppData,
	CudaContext& context) {

	std::auto_ptr<SegCsrPreprocessData> data(new SegCsrPreprocessData);

	int2 launch = Tuning::GetLaunchParams(context);
	int NV = launch.x * launch.y;

	int numTiles = SGPU_DIV_UP(count, NV);
	data->count = count;
	data->numSegments = data->numSegments2 = numSegments;
	data->numTiles = numTiles;

	// Filter out empty rows and build a replacement structure.
	if(supportEmpty) {
		SGPU_MEM(int) csr2Device = context.Malloc<int>(numSegments + 1);
		CsrStripEmpties<false>(count, csr_global, (const int*)0, numSegments,
			csr2Device->get(), (int*)0, (int*)&data->numSegments2, context);
		if(data->numSegments2 < numSegments) {
			csr_global = csr2Device->get();
			numSegments = data->numSegments2;
			data->csr2Device = csr2Device;
		}
	}

	data->limitsDevice = PartitionCsrPlus(count, NV, csr_global, numSegments,
		(const int*)0, numTiles + 1, context);
	data->threadCodesDevice = BuildCsrPlus<Tuning>(count, csr_global,
		data->limitsDevice->get(), numTiles, context);

	*ppData = data;
}

////////////////////////////////////////////////////////////////////////////////
// CsrBulkInsert
// A specialized version of BulkInsert which reads the insertion count from
// global memory rather than as a kernel argument--this lets us avoid a costly
// device->host copy after running CsrStripEmpties.

// Not tolerant to large block sizes, but do not expect them.
template<int NT>
__global__ void KernelCsrBulkInsertPartition(const int* csr2_global,
	int numRows, int* mp_global, int numSearches, int nv) {

	int partition = NT * blockIdx.x + threadIdx.x;
	int numRows2 = csr2_global[numRows];
	int aCount = numRows - numRows2;
	int bCount = numRows2;

	const int* indices_global = csr2_global + numRows2;

	if(partition < numSearches) {
		int diag = nv * partition;
		int mp = MergePath<SgpuBoundsLower>(indices_global, aCount,
			sgpu::counting_iterator<int>(0), bCount, min(diag, numRows),
			sgpu::less<int>());
		mp_global[partition] = mp;
	}
}

template<typename Tuning, typename T, typename OutputIt>
SGPU_LAUNCH_BOUNDS void KernelCsrBulkInsertSpecial(int numTiles,
	const int* csr2_global,	const T* data_global, int numRows,
	const int* mp_global, T identity, OutputIt dest_global) {

	typedef SGPU_LAUNCH_PARAMS Params;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;

	union Shared {
		typename CTABulkInsert<NT, VT>::Storage storage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;

	for(int block = blockIdx.x; block < numTiles; block += gridDim.x) {
		__syncthreads();

		int gid = NV * block;
		int count2 = min(NV, numRows - gid);

		int numRows2 = csr2_global[numRows];
		int4 range = ComputeMergeRange(numRows, 0, block, 0, NV, mp_global);

		int aCount = numRows - numRows2;

		const int* indices_global = csr2_global + numRows2;
		CTABulkInsert<NT, VT>::BuildGatherIndices(range, indices_global, tid,
			shared.storage);

		int indices[VT];
		DeviceSharedToReg<NT, VT>(shared.storage.indices, tid, indices);

		int b0 = range.z;		// B is source array offset.
		aCount = range.y - range.x;

		data_global += b0 - aCount;
		T values[VT];
		#pragma unroll
		for(int i = 0; i < VT; ++i) {
			int index = indices[i];
			if(index < aCount) values[i] = identity;
			else if(index < count2) values[i] = data_global[index];
		}

		DeviceRegToGlobal<NT, VT>(count2, values, tid, dest_global + gid);
	}
}

template<typename T, typename DestIt>
SGPU_HOST void CsrBulkInsert(const int* csr2_global, int numRows,
	const T* data_global, T identity, DestIt dest_global,
	CudaContext& context) {

	typedef LaunchBoxVT<
		128, 3, 0,
		128, 3, 0,
		128, 3, 0
	> Tuning;
	int2 launch = Tuning::GetLaunchParams(context);
	int NV = launch.x * launch.y;

	int numTiles = SGPU_DIV_UP(numRows, NV);
	int numPartitions = numTiles + 1;
	int numPartitionBlocks = SGPU_DIV_UP(numPartitions, 64);

	// Run a Merge Path partitioning to divide the data over equal intervals.
	SGPU_MEM(int) partitionsDevice = context.Malloc<int>(numPartitions);
	KernelCsrBulkInsertPartition<64>
		<<<numPartitionBlocks, 64, 0, context.Stream()>>>(csr2_global,
		numRows, partitionsDevice->get(), numPartitions, NV);
	SGPU_SYNC_CHECK("KernelCsrBulkInsertPartition");

	// Launch the special Csr BulkInsert kernel to plug the empty rows with
	// the identity element.
	int maxBlocks = context.MaxGridSize();
	int numBlocks = min(numTiles, maxBlocks);
	KernelCsrBulkInsertSpecial<Tuning>
		<<<numBlocks, launch.x, 0, context.Stream()>>>(numTiles, csr2_global,
		data_global, numRows, partitionsDevice->get(), identity, dest_global);
	SGPU_SYNC_CHECK("KernelCsrBulkInsertSpecial");
}

} // namespace sgpu
