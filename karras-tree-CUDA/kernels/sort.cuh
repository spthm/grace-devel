#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

namespace grace {

template <typename UInteger, typename Ta, typename Tb>
void sort_by_key(thrust::device_vector<UInteger>& d_keys,
                 thrust::device_vector<Ta>& d_a,
                 thrust::device_vector<Tb>& d_b)
{
    thrust::device_vector<UInteger> d_keys2 = d_keys;

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_a.begin());

    thrust::sort_by_key(d_keys2.begin(), d_keys2.end(), d_b.begin());
}

template <typename UInteger, typename Ta, typename Tb>
void sort_by_key(thrust::host_vector<UInteger>& h_keys,
                 thrust::host_vector<Ta>& h_a,
                 thrust::host_vector<Tb>& h_b)
{
    thrust::host_vector<UInteger> h_keys2 = h_keys;

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_a.begin());

    thrust::sort_by_key(h_keys2.begin(), h_keys2.end(), h_b.begin());
}

template <typename Float, typename Float4>
class particle_distance_functor
{
    const float3 origin;
    const Float4* particles;
    Float4 lpar, rpar;
    Float l, r;

public:
    particle_distance_functor(float3 _origin, Float4* _particles) :
        origin(_origin), particles(_particles) {}

    __host__ __device__ bool operator() (unsigned int li, unsigned int ri)
    {
        lpar = particles[li];
        rpar = particles[ri];

        l = (lpar.x - origin.x)*(lpar.x - origin.x)
             + (lpar.y - origin.y)*(lpar.y - origin.y)
             + (lpar.z - origin.z)*(lpar.z - origin.z);
        r = (rpar.x - origin.x)*(rpar.x - origin.x)
             + (rpar.y - origin.y)*(rpar.y - origin.y)
             + (rpar.z - origin.z)*(rpar.z - origin.z);

        // Distance along a ray will always be positive, no need to sqrt.
        return l < r;
    }
};

template <typename Float, typename Float4, typename T>
void sort_by_distance(const Float origin_x,
                      const Float origin_y,
                      const Float origin_z,
                      const thrust::device_vector<Float4>& d_particles,
                      const thrust::device_vector<unsigned int>& d_ray_offsets,
                      thrust::device_vector<unsigned int>& d_hit_indices,
                      thrust::device_vector<T>& d_hit_data)
{
    size_t total_hits = d_hit_indices.size();
    size_t n_rays = d_ray_offsets.size();

    float3 origin = make_float3(origin_x, origin_y, origin_z);

    thrust::device_vector<unsigned int> d_ray_segments(d_hit_indices.size());
    thrust::constant_iterator<unsigned int> first(1);
    thrust::constant_iterator<unsigned int> last = first + n_rays;

    // offsets = [0, 3, 3, 7]
    // scatter:
    //    => ray_segments = [1, 0, 0, 1, 0, 0, 0, 1]
    // inclusive_scan:
    //    => ray_segments = [1, 1, 1, 2, 2, 2, 2, 3]
    thrust::scatter(first, last,
                    d_ray_offsets.begin(),
                    d_ray_segments.begin());
    thrust::inclusive_scan(d_ray_segments.begin(), d_ray_segments.end(),
                           d_ray_segments.begin());

    // Sort the hits by their distance from the source.
    // The custom comparison function means this will be a merge sort, not the
    // faster radix sort, but we could pass d_ray_segments into the functor and
    // use it during the comparison to eliminate the second sort.
    thrust::sort_by_key(d_hit_indices.begin(), d_hit_indices.end(),
                        thrust::make_zip_iterator(
                            thrust::make_tuple(d_hit_data.begin(),
                                               d_ray_segments.begin())
                        ),
                        particle_distance_functor<Float, Float4>(
                            origin,
                            (Float4*)thrust::raw_pointer_cast(d_particles.data())
                        )
    );
    // Sort the hits by their ray ID.  Since this is a stable sort, all the
    // elements belonging to a particular ray ID remain sorted by distance!
    thrust::stable_sort_by_key(d_ray_segments.begin(), d_ray_segments.end(),
                               thrust::make_zip_iterator(
                                   thrust::make_tuple(d_hit_data.begin(),
                                                      d_hit_indices.begin())
                               )
    );
}

} // namespace grace
