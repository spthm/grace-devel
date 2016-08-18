// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "intersection.h"

#include "grace/cuda/generate_rays.cuh"
#include "grace/generic/bits.h"
#include "grace/generic/intersect.h"
#include "grace/ray.h"
#include "grace/sphere.h"
#include "helper/random.cuh"

#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <cstdlib>
#include <iostream>
#include <gmpxx.h>

typedef grace::Sphere<float> SphereType;

struct expand_functor
{
    float d;

    __host__ __device__ expand_functor(float distance): d(distance) {}

    __host__ __device__ SphereType operator()(const SphereType& sphere)
    {
        SphereType s = sphere;
        // Centre assumed (0, 0).
        s.x += d * grace::sgn(s.x);
        s.y += d * grace::sgn(s.y);
        s.z += d * grace::sgn(s.z);
        return s;
    }
};

int main(int argc, char* argv[])
{
    size_t N = 2000000;
    size_t N_rays = 32 * 1000; // = 32,000
    // Error tolerance as fractional error in the square of the normalized
    // impact parameter, measured against reference value from GMP library.
    double tolerance = 1E-8;
    // Defaults to true because it may take hours to re-run in the case of a
    // failure.
    bool verbose = true;

    if (argc > 1) {
        N = (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = 32 * (size_t)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        tolerance = std::atof(argv[3]);
    }
    if (argc > 4) {
        verbose = (std::string(argv[4]) == "true") ? true : false;
    }

    std::cout << "Number of particles: " << N << std::endl
              << "Number of rays:      " << N_rays << std::endl
              << "Error tolerance:     " << tolerance << std::endl
              << std::endl;


    SphereType low = SphereType(-1E4f, -1E4f, -1E4f, 80.f);
    SphereType high = SphereType(1E4f, 1E4f, 1E4f, 400.f);
    thrust::host_vector<SphereType> h_spheres(N);
    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      h_spheres.begin(),
                      random_sphere_functor<SphereType>(low, high));

    // Reference intersection function does not discriminate around start
    // (and end) points of the ray, but GRACE's sphere intersection function
    // does. To avoid this inconsistency, we ensure no particles contain the
    // rays' common origin (0, 0).
    thrust::transform(h_spheres.begin(), h_spheres.end(), h_spheres.begin(),
                      expand_functor(high.r));

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    grace::uniform_random_rays(d_rays, 0.f, 0.f, 0.f, 2E4f);
    thrust::host_vector<grace::Ray> h_rays = d_rays;
    d_rays.clear(); d_rays.shrink_to_fit();

    size_t failures = 0;
    #pragma omp parallel for
    for (size_t ri = 0; ri < N_rays; ++ri)
    {
        grace::Ray ray = h_rays[ri];

        for (size_t si = 0; si < N;  ++si)
        {
            SphereType sphere = h_spheres[si];
            float b2, d;
            bool hit = grace::sphere_hit(ray, sphere, b2, d);
            bool ref_hit = ray_sphere_intersection<mpq_class>(ray, sphere);

            if (ref_hit != hit)
            {
                double ref_b2 = impact_parameter2(ray, sphere);
                double R2 = static_cast<double>(sphere.r) * sphere.r;

                if (std::abs(1.0 - ref_b2 / R2) > tolerance)
                {
                    ++failures;

                    if (!verbose) {
                      continue;
                    }

                    std::cout << "FAILED intersection for ray " << ri
                              << " and sphere " << si << ":" << std::endl
                              << "  Reference result: "
                              << (ref_hit ? "TRUE" : "FALSE") << std::endl
                              << "  GRACE result:     "
                              << (hit ? "TRUE" : "FALSE") << std::endl
                              << "  Reference squared impact parameter: "
                              << ref_b2 << std::endl
                              << "  GRACE squared impact parameter:     "
                              << b2 << std::endl
                              << "  Sphere squared radius:              "
                              << R2 << std::endl
                              << std::endl;
                }
            }
        }
    }

    if (failures == 0)
    {
        std::cout << "PASSED" << std::endl;
    }
    else
    {
        std::cout << "FAILED" << std::endl
                  << failures << " intersection" << (failures > 1 ? "s " : " ")
                  << "tests did not agree with host results" << std::endl;
    }

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
