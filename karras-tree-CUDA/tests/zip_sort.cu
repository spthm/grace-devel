#include <iostream>
#include <cstdlib>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>

#include "../types.h"
#include "../kernels/build_sph.cuh"

__host__ __device__ unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

class random_pos_functor
{
    const int offset;

public:
    random_pos_functor(const int offset_) : offset(offset_) {}

    __host__ __device__ float operator() (unsigned int n)
    {
        unsigned int seed = hash(n);
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<float> u01(0,1);

        rng.discard(offset);

        return u01(rng);
    }
};


int main(int argc, char* argv[]) {

    cudaEvent_t start, stop;
    float elapsed_time;
    float total_time_float4 = 0;
    float total_time_zip = 0;
    float total_time_gather = 0;


    /* Generate N random positions, i.e. 3*N random floats in [0,1) */

    unsigned int N = 1000000;
    unsigned int Niter = 100;
    if (argc > 1)
        N = (unsigned int) std::strtol(argv[1], NULL, 10);
    if (argc > 2)
        Niter = (unsigned int) std::strtol(argv[2], NULL, 10);

    std::cout << "Will generate " << N << " random points for " << Niter
              << " iteration" << ((Niter > 1) ? "s" : "") << "..."
              << std::endl;
    std::cout << std::endl;

    thrust::host_vector<float> h_x_centres(N);
    thrust::host_vector<float> h_y_centres(N);
    thrust::host_vector<float> h_z_centres(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      h_x_centres.begin(),
                      random_pos_functor(0) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      h_y_centres.begin(),
                      random_pos_functor(1) );
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      h_z_centres.begin(),
                      random_pos_functor(2) );


    /* Copy centres into a host vector of float4s. */

    thrust::host_vector<float4> h_centres4(N);
    for (int i=0; i<N; i++) {
        h_centres4[i].x = h_x_centres[i];
        h_centres4[i].y = h_y_centres[i];
        h_centres4[i].z = h_z_centres[i];
        h_centres4[i].w = 0.0f;
    }


    /* Generate the Morton key of each position. */

    thrust::device_vector<float4> d_centres4 = h_centres4;
    thrust::device_vector<grace::uinteger32> d_keys(N);
    float3 bottom = make_float3(0., 0., 0.);
    float3 top = make_float3(1., 1., 1.);

    grace::morton_keys_sph(d_centres4, top, bottom, d_keys);

    thrust::host_vector<float> h_keys = d_keys;


    /* Sort the centres vector, and the separate x, y, z vectors using a zip
     * iterator and an indices + gather method.  Repeat and record total time
     * taken for each method.
     */


    /* Measure time for sorting the float4. */

    std::cout << "Running float4 sort iterations..." << std::endl;
    for (int i=0; i<Niter; i++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_centres4.begin());

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_time_float4 += elapsed_time;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        d_centres4 = h_centres4;
        d_keys = h_keys;
    }


    /* Measure time for sorting the  zip iterator. */

    thrust::device_vector<float> d_x_centres = h_x_centres;
    thrust::device_vector<float> d_y_centres = h_y_centres;
    thrust::device_vector<float> d_z_centres = h_z_centres;
    std::cout << "Running zip iterator sort iterations..." << std::endl;
    for (int i=0; i<Niter; i++) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        thrust::sort_by_key(d_keys.begin(),
                            d_keys.end(),
                            thrust::make_zip_iterator(
                                thrust::make_tuple(d_x_centres.begin(),
                                                   d_y_centres.begin(),
                                                   d_z_centres.begin() ))
                            );

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_time_zip += elapsed_time;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        d_x_centres = h_x_centres;
        d_y_centres = h_y_centres;
        d_z_centres = h_z_centres;
        d_keys = h_keys;
    }


    /* Measure time for sorting the gather method. */

    std::cout << "Running index sort and gather iterations..." << std::endl;
    thrust::device_vector<int> d_indices(N);
    thrust::device_vector<float> d_tmp(N);
    for (int i=0; i<Niter; i++) {
        thrust::sequence(d_indices.begin(), d_indices.end());

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_indices.begin());

        thrust::gather(d_indices.begin(),
                       d_indices.end(),
                       d_x_centres.begin(),
                       d_tmp.begin());
        d_x_centres = d_tmp;

        thrust::gather(d_indices.begin(),
                       d_indices.end(),
                       d_y_centres.begin(),
                       d_tmp.begin());
        d_y_centres = d_tmp;

        thrust::gather(d_indices.begin(),
                       d_indices.end(),
                       d_z_centres.begin(),
                       d_tmp.begin());
        d_z_centres = d_tmp;

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsed_time, start, stop);
        total_time_gather += elapsed_time;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        d_x_centres = h_x_centres;
        d_y_centres = h_y_centres;
        d_z_centres = h_z_centres;
        d_keys = h_keys;
    }

    std::cout << "Mean time taken for float4:             "
              << total_time_float4 / (float) Niter << " ms." << std::endl;
    std::cout << "Mean time taken for zip iterator:       "
              << total_time_zip / (float) Niter << " ms." << std::endl;
    std::cout << "Mean time taken for indices and gather: "
              << total_time_gather / (float) Niter << " ms." << std::endl;

}
