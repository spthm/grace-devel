#pragma once

#include "grace/error.h"
#include "grace/types.h"

#include "grace/external/sgpu/kernels/segrscancsr.cuh"

namespace grace {

template <typename Real>
GRACE_HOST void exclusive_segmented_scan(
    const thrust::device_vector<int>& d_segment_offsets,
    thrust::device_vector<Real>& d_data,
    thrust::device_vector<Real>& d_results)
{
    // SGPU calls require a context.
    int device_ID = 0;
    GRACE_CUDA_CHECK(cudaGetDevice(&device_ID));
    sgpu::ContextPtr sgpu_context_ptr = sgpu::CreateCudaDevice(device_ID);

    std::auto_ptr<sgpu::SegCsrPreprocessData> pp_data_ptr;

    // TODO: Perform the segmented scan in-place (sgpu::SegScanApplyExc).
    size_t N_data = d_data.size();
    size_t N_segments = d_segment_offsets.size();

    sgpu::SegScanCsrPreprocess<Real>(
        N_data, thrust::raw_pointer_cast(d_segment_offsets.data()),
        N_segments, true, &pp_data_ptr, *sgpu_context_ptr);

    sgpu::SegScanApply<sgpu::SgpuScanTypeExc>(*pp_data_ptr,
        thrust::raw_pointer_cast(d_data.data()), Real(0), sgpu::plus<Real>(),
        thrust::raw_pointer_cast(d_results.data()), *sgpu_context_ptr);
}

} // namespace grace
