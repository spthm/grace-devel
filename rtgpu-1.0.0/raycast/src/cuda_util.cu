#include <cudpp/cudpp.h>
#include <cassert>
#include <iostream>
#include "dvector.h"
#include "util.h"

#define USE_THRUST 0

#if USE_THRUST
#include <thrust/detail/raw_buffer.h>
#include <thrust/scan.h>

struct head_flags_pred
{
    template <class T>
    __host__ __device__ bool operator()(const T &, const T &b) const
    {
        return b ? false : true;
    }
};

#endif

#if 0
class init_cudpp
{
public:
    init_cudpp()
    {
        CUDPPResult res = cudppCreate(&m_handle);
        if(res != CUDPP_SUCCESS)
            throw std::runtime_error("Error initializing CUDPP library");
    }
    ~init_cudpp()
    {
        cudppDestroy(m_handle);
    }

    operator CUDPPHandle() const { return m_handle; }

private:
    CUDPPHandle m_handle;
} theCudpp;
#endif

void compute_linear_grid(unsigned size, dim3 &grid, dim3 &block)
{
    int g, b;

    b = 256;

    g = (size+b-1)/b;
#if 0
    if(g > 30 && b >= 64)
    {
        b -= 32;
        g = (size+b-1)/b;
    }
#endif

    grid = dim3(g);
    block = dim3(b);
}

__host__ void segscan_add(dvector<unsigned> &dest, 
                          const dvector<unsigned> &orig,
                          const dvector<unsigned> &head_flags)
{
    assert(orig.size() == head_flags.size());
    dest.resize(orig.size());

#if USE_THRUST
    thrust::device_ptr<const unsigned> porig(orig.data()), 
                                       pflags(head_flags.data());
    thrust::device_ptr<unsigned> pdest(dest.data());

    inclusive_scan_by_key(pflags, pflags+head_flags.size(),
                          porig, pdest, head_flags_pred());
#else


    static CUDPPHandle plan = CUDPP_INVALID_HANDLE;
    static size_t max_size = 0;
    static size_t use_count = 0;

    struct destroy_plan
    {
        static void call()
        {
            cudppDestroyPlan(plan);
            plan = CUDPP_INVALID_HANDLE;
        }
    };

    if(use_count++==10 || plan == CUDPP_INVALID_HANDLE || max_size<orig.size())
    {
        if(plan != CUDPP_INVALID_HANDLE)
            destroy_plan::call();

        CUDPPConfiguration config;
        config.op = CUDPP_ADD;
        config.datatype = CUDPP_UINT;
        config.algorithm = CUDPP_SEGMENTED_SCAN;
        config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;

        CUDPPResult res = cudppPlan(&plan, config, orig.size(), 1, 0);
        if(res != CUDPP_SUCCESS)
            throw std::runtime_error("Error creating CUDPPPlan");

        static bool atexit_set = false;
        if(!atexit_set)
        {
            atexit(&destroy_plan::call);
            atexit_set = true;
        }

        max_size = orig.size();
    }

    assert(cudaThreadSynchronize() == cudaSuccess);

    CUDPPResult res = 
        cudppSegmentedScan(plan, dest, orig,head_flags, orig.size());

    assert(cudaThreadSynchronize() == cudaSuccess);
    
    if(res != CUDPP_SUCCESS)
        throw std::runtime_error("Error during segmented scan");
#endif
}


__host__ void scan_add(dvector<unsigned> &dest, 
                       const dvector<unsigned> &orig, 
                       scan_type type)
{
    CUDPPHandle cur_plan;

    if(type == INCLUSIVE)
    {
        static CUDPPHandle plan = CUDPP_INVALID_HANDLE;
        static size_t max_size = 0;
        static size_t use_count = 0;

        struct destroy_plan
        {
            static void call()
            {
                cudppDestroyPlan(plan);
                plan = CUDPP_INVALID_HANDLE;
            }
        };

        if(use_count++==10 || plan == CUDPP_INVALID_HANDLE || max_size<orig.size())
        {
            if(plan != CUDPP_INVALID_HANDLE)
                destroy_plan::call();

            CUDPPConfiguration config;
            config.op = CUDPP_ADD;
            config.datatype = CUDPP_UINT;
            config.algorithm = CUDPP_SCAN;
            config.options = CUDPP_OPTION_FORWARD;
            if(type == INCLUSIVE)
                config.options |= CUDPP_OPTION_INCLUSIVE;
            else 
                config.options |= CUDPP_OPTION_EXCLUSIVE;

            CUDPPResult res = cudppPlan(&plan, config, orig.size(), 1, 0);
            if(res != CUDPP_SUCCESS)
                throw std::runtime_error("Error creating CUDPPPlan");

            static bool atexit_set = false;
            if(!atexit_set)
            {
                atexit(&destroy_plan::call);
                atexit_set = true;
            }

            max_size = orig.size();
        }

        cur_plan = plan;
    }
    else // exclusive
    {
        static CUDPPHandle plan = CUDPP_INVALID_HANDLE;
        static size_t max_size = 0;
        static size_t use_count = 0;

        struct destroy_plan
        {
            static void call()
            {
                cudppDestroyPlan(plan);
                plan = CUDPP_INVALID_HANDLE;
            }
        };

        if(use_count++==10 || plan == CUDPP_INVALID_HANDLE || max_size<orig.size())
        {
            if(plan != CUDPP_INVALID_HANDLE)
                destroy_plan::call();

            CUDPPConfiguration config;
            config.op = CUDPP_ADD;
            config.datatype = CUDPP_UINT;
            config.algorithm = CUDPP_SCAN;
            config.options = CUDPP_OPTION_FORWARD;
            config.options |= CUDPP_OPTION_EXCLUSIVE;

            CUDPPResult res = cudppPlan(&plan, config, orig.size(), 1, 0);
            if(res != CUDPP_SUCCESS)
                throw std::runtime_error("Error creating CUDPPPlan");

            static bool atexit_set = false;
            if(!atexit_set)
            {
                atexit(&destroy_plan::call);
                atexit_set = true;
            }

            max_size = orig.size();
        }

        cur_plan = plan;
    }

    dest.resize(orig.size());

    CUDPPResult res = cudppScan(cur_plan, dest, orig, orig.size());

    if(res != CUDPP_SUCCESS)
        throw std::runtime_error("Error during scan");
}


__host__ void sort(dvector<unsigned> &keys, dvector<unsigned> &values,
                   int bits)
{
    assert(keys.size() == values.size());

    static CUDPPHandle plan = CUDPP_INVALID_HANDLE;
    static size_t max_size = 0;
    static size_t use_count = 0;
    struct destroy_plan
    {
        static void call()
        {
            cudppDestroyPlan(plan);
            plan = CUDPP_INVALID_HANDLE;
        }
    };


    if(use_count++==10 || plan == CUDPP_INVALID_HANDLE || max_size < keys.size())
    {
        if(plan != CUDPP_INVALID_HANDLE)
            destroy_plan();

        CUDPPConfiguration config;
        config.datatype = CUDPP_UINT;
        config.algorithm = CUDPP_SORT_RADIX;
        config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

        CUDPPResult res = cudppPlan(&plan, config, keys.size(), 1, 0);

        if(res != CUDPP_SUCCESS)
            throw std::runtime_error("Error creating CUDPPPlan");

        static bool atexit_set = false;
        if(!atexit_set)
        {
            atexit(&destroy_plan::call);
            atexit_set = true;
        }

        max_size = keys.size();
    }

    CUDPPResult res = cudppSort(plan, keys, values, bits, keys.size());

    if(res != CUDPP_SUCCESS)
        throw std::runtime_error("Error during radix sort");
}


// adjacent_difference ------------------------------------------------

__global__ void adjacent_difference(const unsigned *input, unsigned *output,
                                    size_t count, size_t input_item_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count)
        return;

    if(idx == count-1)
        output[idx] = input_item_count - input[idx];
    else
        output[idx] = input[idx+1] - input[idx];
}

__host__ void adjacent_difference(dvector<unsigned> &output, 
                                  const dvector<unsigned> &input, 
                                  size_t input_item_count)
{
    output.resize(input.size());


    dim3 dimGrid, dimBlock;
    compute_linear_grid(input.size(), dimGrid, dimBlock);

    adjacent_difference<<<dimGrid, dimBlock>>>(input, output, input.size(),
                                               input_item_count);
}

__host__ void cuda_synchronize()
{
    cudaThreadSynchronize();
    check_cuda_error("Error executing kernel");
}

