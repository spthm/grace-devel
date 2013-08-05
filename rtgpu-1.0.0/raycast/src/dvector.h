#ifndef DVECTOR_H
#define DVECTOR_H

#include <vector>
#include "alloc.h"

template <class T>
class dvector
{
public:
    dvector(const std::vector<T> &that)
        : m_data(NULL)
        , m_size(0)
        , m_capacity(0)
    {
        *this = that;
    }

    dvector(const dvector &that)
        : m_data(NULL)
        , m_size(0)
        , m_capacity(0)
    {
        *this = that;
    }

    dvector(size_t size=0)
        : m_data(NULL)
        , m_size(0)
        , m_capacity(0)
    {
        resize(size);
    }
    
    ~dvector()
    {
        cuda_delete(m_data);
    }

    void resize(size_t size)
    {
        if(size > m_capacity)
        {
            cuda_delete(m_data);
            m_data = NULL;
            m_capacity = 0;
            m_size = 0;

            m_data = cuda_new<T>(size);
            m_capacity = size;
            m_size = size;
        }
        else
            m_size = size;
    }

    T operator[](int idx) const
    {
        T value;
        cudaMemcpy(&value, data()+idx, sizeof(T), cudaMemcpyDeviceToHost);
        return value;
    }

    dvector &operator=(const dvector &that)
    {
        resize(that.size());
        cudaMemcpy(data(), that.data(), size()*sizeof(T), 
                                       cudaMemcpyDeviceToDevice);
        check_cuda_error("Error during memcpy from device to device");
        return *this;
    }

    dvector &operator=(const std::vector<T> &that)
    {
        resize(that.size());
        cudaMemcpy(data(), that.data(), size()*sizeof(T), 
                                       cudaMemcpyHostToDevice);
        check_cuda_error("Error during memcpy from host to device");
        return *this;
    }

    bool empty() const { return size()==0; }
    size_t size() const { return m_size; }

    T *data() { return m_data; }
    const T *data() const { return m_data; }

    T back() const { return operator[](size()-1); }

    operator T*() { return data(); }
    operator const T*() const { return data(); }

    friend void swap(dvector &a, dvector &b)
    {
        std::swap(a.m_data, b.m_data);
        std::swap(a.m_size, b.m_size);
        std::swap(a.m_capacity, b.m_capacity);
    }

private:
    T *m_data;
    size_t m_size, m_capacity;
};

template <class T>
std::vector<T> to_cpu(const dvector<T> &v)
{
    std::vector<T> out;
    out.resize(v.size());

    cudaMemcpy(&out[0], v.data(), v.size()*sizeof(T),
                                 cudaMemcpyDeviceToHost);
    check_cuda_error("Error during memcpy from device to host");

    return out; // hope that RVO will kick in
}



#endif
