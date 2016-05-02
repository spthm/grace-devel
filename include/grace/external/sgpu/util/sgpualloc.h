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

#include "../util/util.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace sgpu {

class CudaDevice;

////////////////////////////////////////////////////////////////////////////////
// Customizable allocator.

// CudaAlloc is the interface class all allocator accesses. Users may derive
// this, implement custom allocators, and set it to the device with
// CudaDevice::SetAllocator.

class CudaAlloc : public CudaBase {
public:
	virtual cudaError_t Malloc(size_t size, void** p) = 0;
	virtual bool Free(void* p) = 0;
	virtual void Clear() = 0;

	virtual ~CudaAlloc() { }

	CudaDevice& Device() { return _device; }

protected:
	CudaAlloc(CudaDevice& device) : _device(device) { }
	CudaDevice& _device;
};

// A concrete class allocator that simply calls cudaMalloc and cudaFree.
class CudaAllocSimple : public CudaAlloc {
public:
	CudaAllocSimple(CudaDevice& device) : CudaAlloc(device) { }

	virtual cudaError_t Malloc(size_t size, void** p);
	virtual bool Free(void* p);
	virtual void Clear() { }
	virtual ~CudaAllocSimple() { }
};


inline cudaError_t CudaAllocSimple::Malloc(size_t size, void** p) {
	cudaError_t error = cudaSuccess;
	*p = 0;
	if(size) error = cudaMalloc(p, size);

	if(cudaSuccess != error) {
		printf("CUDA MALLOC ERROR %d\n", error);
		exit(0);
	}

	return error;
}

inline bool CudaAllocSimple::Free(void* p) {
	cudaError_t error = cudaSuccess;
	if(p) error = cudaFree(p);
	return cudaSuccess == error;
}

} // namespace sgpu
