// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include <cstdlib>
#include <iostream>
#include <gmpxx.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

#include "intersection.cuh"

#include "../../ray.h"
#include "../../utils.cuh"
#include "../../device/bits.cuh"
#include "../../device/intersect.cuh"
#include "../../kernels/gen_rays.cuh"

struct expand_functor
{
    float d;

    __host__ __device__ expand_functor(float distance): d(distance) {}

    __host__ __device__ float4 operator()(const float4& sphere)
    {
        float4 s = sphere;
        // Centre assumed (0, 0).
        s.x += d * grace::bits::sgn(s.x);
        s.y += d * grace::bits::sgn(s.y);
        s.z += d * grace::bits::sgn(s.z);
        return s;
    }
};

int main(int argc, char* argv[]) {

    /* Initialize run parameters. */

    size_t N = 2000000;
    size_t N_rays = 32 * 1000; // = 32,000
    double tolerance = 1E-8;

    if (argc > 1) {
        N = (size_t) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = 32 * (size_t) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        tolerance = std::atof(argv[3]);
    }

    std::cout << "Testing " << N << " random points and " << N_rays
              << " random rays." << std::endl;
    std::cout << "Error tolerance of " << tolerance
              << " in square of normalized impact parameter." << std::endl;
    std::cout << std::endl;


    /* Generate random points. */

    float min_radius = 80.f;
    float max_radius = 400.f;
    thrust::host_vector<float4> h_spheres(N);
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      h_spheres.begin(),
                      grace::random_float4_functor(-1E4f, 1E4f,
                                                   min_radius, max_radius));

    // Reference intersection function does not discriminate around start
    // (and end) points of the ray, but GRACE's sphere intersection function
    // does. To avoid this inconsistency, ensure no particles contain the ray's
    // origin.
    thrust::transform(h_spheres.begin(), h_spheres.end(),
                      h_spheres.begin(),
                      expand_functor(max_radius));


    /* Generate the rays (emitted from box centre and of length 2E4). */

    float ox, oy, oz, length;
    ox = oy = oz = 0.0f;
    length = 2E4f;

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    grace::uniform_random_rays(d_rays, ox, oy, oz, length);
    thrust::host_vector<grace::Ray> h_rays = d_rays;
    d_rays.clear(); d_rays.shrink_to_fit();

    /* Loop through all rays and test for interestion with all particles
     * directly. When reference intersection test does not agree with GRACE's
     * intersection test (subject to tolerance), print as a failure.
     */

    size_t failures = 0;
    #pragma omp parallel for
    for (size_t ri = 0; ri < N_rays; ++ri)
    {
        grace::Ray ray = h_rays[ri];

        for (size_t si = 0; si < N;  ++si)
        {
            float4 sphere = h_spheres[si];
            float b2, d;
            bool hit = grace::sphere_hit(ray, sphere, b2, d);
            bool ref_hit = ray_sphere_intersection<mpq_class>(ray, sphere);

            if (ref_hit != hit)
            {
                double ref_b2 = impact_parameter2(ray, sphere);
                double R2 = static_cast<double>(sphere.w) * sphere.w;

                if (std::abs(1.0 - ref_b2 / R2) > tolerance) {
                    ++failures;
                    std::cout << "FAILED intersection for ray " << ri
                              << " and sphere " << si << ":" << std::endl;
                    std::cout << "  Reference result: "
                              << (ref_hit ? "TRUE" : "FALSE") << std::endl;
                    std::cout << "  GRACE result:     "
                              << (hit ? "TRUE" : "FALSE") << std::endl;
                    std::cout << "  Reference squared impact parameter: "
                              << ref_b2 << std::endl;
                    std::cout << "  GRACE squared impact parameter:     "
                              << b2 << std::endl;
                    std::cout << "  Sphere squared radius:              "
                              << R2 << std::endl;
                    std::cout << std::endl;
                }
            }
        }
    }

    if (failures == 0)
    {
        std::cout << "PASSED"
                  << std::endl
                  << "All GRACE ray-sphere intersection tests agree with "
                  << "reference implementation." << std::endl;
    }
    else
    {
        std::cout << "FAILED for " << failures << " intersection(s)."
                  << std::endl;
    }

    return EXIT_SUCCESS;
}
