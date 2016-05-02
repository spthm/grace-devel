/******************************************************************************
 * Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
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
 * Code and text by Sean Baxter, NVIDIA Research
 * See http://nvlabs.github.io/moderngpu for repository and documentation.
 *
 ******************************************************************************/

#pragma once

#include "../sgpuhost.cuh"
#include "../device/ctareduce.cuh"

namespace sgpu {

////////////////////////////////////////////////////////////////////////////////
// KernelReduce

template<typename Tuning, typename InputIt, typename T, typename Op>
SGPU_LAUNCH_BOUNDS void KernelReduce(int numTiles, InputIt data_global,
	int count, T identity, Op op, T* reduction_global) {

	typedef SGPU_LAUNCH_PARAMS Params;
	typedef typename Op::first_argument_type OpT;
	const int NT = Params::NT;
	const int VT = Params::VT;
	const int NV = NT * VT;
	typedef CTAReduce<NT, OpT, Op> R;

	union Shared {
		typename R::Storage reduceStorage;
	};
	__shared__ Shared shared;

	int tid = threadIdx.x;

	for(int block = blockIdx.x; block < numTiles; block += gridDim.x) {
		__syncthreads();

		int gid = NV * block;
		int count2 = min(NV, count - gid);

		// Load a full tile into register in strided order. Set out-of-range values
		// with identity.
		OpT data[VT];
		DeviceGlobalToRegDefault<NT, VT>(count2, data_global + gid, tid, data,
			identity);

		// Sum elements within each thread.
		OpT x;
		#pragma unroll
		for(int i = 0; i < VT; ++i)
			x = i ? op(x, data[i]) : data[i];

		// Sum thread-totals over the CTA.
		x = R::Reduce(tid, x, shared.reduceStorage, op);

		// Store the tile's reduction to global memory.
		if(!tid)
			reduction_global[block] = x;
	}
}

////////////////////////////////////////////////////////////////////////////////
// Reduce

template<typename InputIt, typename T, typename Op>
SGPU_HOST void Reduce(InputIt data_global, int count, T identity, Op op,
	T* reduce_global, T* reduce_host, CudaContext& context) {

	SGPU_MEM(T) totalDevice;
	if(!reduce_global) {
		totalDevice = context.Malloc<T>(1);
		reduce_global = totalDevice->get();
	}

	if(count <= 256) {
		typedef LaunchBoxVT<256, 1> Tuning;
		KernelReduce<Tuning><<<1, 256, 0, context.Stream()>>>(
			1, data_global, count, identity, op, reduce_global);
		SGPU_SYNC_CHECK("KernelReduce");

	} else if(count <= 768) {
		typedef LaunchBoxVT<256, 3> Tuning;
		KernelReduce<Tuning><<<1, 256, 0, context.Stream()>>>(
			1, data_global, count, identity, op, reduce_global);
		SGPU_SYNC_CHECK("KernelReduce");

	} else if(count <= 512 * ((sizeof(T) > 4) ? 4 : 8)) {
		typedef LaunchBoxVT<512, (sizeof(T) > 4) ? 4 : 8> Tuning;
		KernelReduce<Tuning><<<1, 512, 0, context.Stream()>>>(
			1, data_global, count, identity, op, reduce_global);
		SGPU_SYNC_CHECK("KernelReduce");

	} else {
		// Launch a grid and reduce tiles to temporary storage.
		typedef LaunchBoxVT<256, (sizeof(T) > 4) ? 8 : 16> Tuning;
		int2 launch = Tuning::GetLaunchParams(context);
		int NV = launch.x * launch.y;
		int numTiles = SGPU_DIV_UP(count, NV);
		int maxBlocks = context.MaxGridSize();
		int numBlocks = min(numTiles, maxBlocks);

		SGPU_MEM(T) reduceDevice = context.Malloc<T>(numTiles);
		KernelReduce<Tuning><<<numBlocks, launch.x, 0, context.Stream()>>>(
			numTiles, data_global, count, identity, op, reduceDevice->get());
		SGPU_SYNC_CHECK("KernelReduce");

		Reduce(reduceDevice->get(), numTiles, identity, op, reduce_global,
			(T*)0, context);
	}

	if(reduce_host)
		copyDtoH(reduce_host, reduce_global, 1);
}

template<typename InputIt>
SGPU_HOST typename std::iterator_traits<InputIt>::value_type
Reduce(InputIt data_global, int count, CudaContext& context) {
	typedef typename std::iterator_traits<InputIt>::value_type T;
	T result;
	Reduce(data_global, count, (T)0, sgpu::plus<T>(), (T*)0, &result, context);
	return result;
}

} // namespace sgpu
