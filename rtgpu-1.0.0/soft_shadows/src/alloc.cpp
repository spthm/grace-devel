#include "pch.h"
#include "alloc.h"

#define TRACE 0

cuda_memory_pool cuda_memory;

struct cuda_memory_pool::impl
{
    typedef boost::bimap<void *, boost::bimaps::multiset_of<size_t>> buffer_map;

    buffer_map buffers;
    std::unordered_map<void *, size_t> used_buffers;

    size_t total_reserved, total_available;
};

cuda_memory_pool::cuda_memory_pool()
    : pimpl(new impl)
{
    pimpl->total_reserved = 0;
    pimpl->total_available = 0;
}

cuda_memory_pool::~cuda_memory_pool()
{
    recycle();

    for(auto it=pimpl->used_buffers.begin();it!=pimpl->used_buffers.end();++it)
        cudaFree(it->first);

    delete pimpl;
}

void cuda_memory_pool::recycle()
{
    size_t orig_size = pimpl->total_reserved;

    for(auto it=pimpl->buffers.begin(); it!=pimpl->buffers.end(); ++it)
    {
        cudaFree(it->left);
        pimpl->total_reserved -= it->right;
    }

    pimpl->buffers.clear();
    pimpl->total_available = 0;

#if TRACE
    std::cout << "RECYCLE freed " << orig_size-pimpl->total_reserved 
              << " bytes" << std::endl;
#endif
}

void *cuda_memory_pool::malloc(size_t size)
{
    auto it = pimpl->buffers.right.lower_bound(size);

    void *mem = NULL;

    if(it == pimpl->buffers.right.end() || it->first - size >= 1024*1024)
    {
        if(pimpl->total_available > 100*1024*1024)
            recycle();

        cudaMalloc(&mem, size);
        if(mem == NULL)
        {
            recycle();
            cudaMalloc(&mem, size);
            if(mem == NULL)
            {
                std::stringstream ss;
                ss << "Error allocating " << size << " bytes on cuda device";
                throw std::runtime_error(ss.str());
            }
        }
        pimpl->total_reserved += size;

#if TRACE
        std::cout << "MALLOC " << size << " bytes";
        if(it != pimpl->buffers.right.end())
            std::cout << ", cannot use " << it->first << " bytes buffer";
        std::cout << std::endl;
#endif
    }
    else
    {
#if TRACE
        std::cout << "REUSING " << size << "/" << it->first << " bytes " << std::endl;
#endif

        pimpl->total_available -= it->first;

        mem = it->second;
        size = it->first;
        pimpl->buffers.right.erase(it);
    }

    pimpl->used_buffers.insert({mem, size});

    return mem;
}

void cuda_memory_pool::free(void *ptr)
{
    auto it = pimpl->used_buffers.find(ptr);
    if(it == pimpl->used_buffers.end())
        throw std::runtime_error("Memory deallocation error");

#if TRACE
    std::cout << "RETURNING " << it->second << " bytes" << std::endl;
#endif

    pimpl->total_available += it->second;

    pimpl->buffers.insert({it->first, it->second});
    pimpl->used_buffers.erase(it);
}
