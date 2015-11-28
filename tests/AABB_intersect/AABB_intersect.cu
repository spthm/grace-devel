#include "AABB.cuh"
#include "ray.cuh"
#include "profile.cuh"
#include "intersectors.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[])
{
    unsigned int N_rays = 100000;
    unsigned int N_AABBs = 5000;
    bool verbose = true;

    if (argc > 1)
        N_rays = (unsigned int)std::strtol(argv[1], NULL, 10);
    if (argc > 2)
        N_AABBs = (unsigned int)std::strtol(argv[2], NULL, 10);
    if (argc > 3)
        verbose = std::string(argv[3]) == "true" ? true : false;

    std::cout << "Testing " << N_rays << " rays against "
              << N_AABBs << " AABBs." << std::endl;
    std::cout << std::endl;


    thrust::host_vector<Ray> h_rays(N_rays);
    thrust::host_vector<AABB> h_AABBs(N_AABBs);

    // Generate isotropic rays from the origin, and of length 2.
    isotropic_rays(h_rays, 0.f, 0.f, 0.f, 2.f);

    // Generate the AABBs, with all points uniformly random in [-1, 1).
    random_aabbs(h_AABBs, -1.f, 1.f);

    float elapsed;
    thrust::device_vector<Ray> d_rays = h_rays;
    thrust::device_vector<AABB> d_AABBs = h_AABBs;
    thrust::device_vector<int> d_hits(N_rays);
    thrust::host_vector<int> h_hits(N_rays);

    elapsed = profile_gpu(d_rays, d_AABBs, d_hits, Aila());
    thrust::host_vector<int> h_aila_laine_hits = d_hits;
    std::cout << "Aila, Laine and Karras:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;


    elapsed = profile_gpu(d_rays, d_AABBs, d_hits, Williams());
    thrust::host_vector<int> h_williams_hits = d_hits;
    std::cout << "Williams:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;


    elapsed = profile_gpu(d_rays, d_AABBs, d_hits, Williams_noif());
    thrust::host_vector<int> h_williams_noif_hits = d_hits;

    std::cout << "Williams (no if):" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;


    elapsed = profile_gpu(d_rays, d_AABBs, d_hits, Eisemann());
    thrust::host_vector<int> h_eisemann_hits = d_hits;
    std::cout << "Eisemann:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;
    // And on CPU.
    elapsed = profile_cpu(h_rays, h_AABBs, h_hits, Eisemann());
    std::cout << "    CPU: " << elapsed << " ms." << std::endl;


    elapsed = profile_gpu(d_rays, d_AABBs, d_hits, Plucker());
    thrust::host_vector<int> h_plucker_hits = d_hits;
    std::cout << "Plucker:" << std::endl;
    std::cout << "    GPU: " << elapsed << " ms." << std::endl;
    // And on CPU.
    elapsed = profile_cpu(h_rays, h_AABBs, h_hits, Plucker());
    std::cout << "    CPU: " << elapsed << " ms." << std::endl;

    std::cout << std::endl;

    compare_hitcounts(h_eisemann_hits, "Eisemann",
                      h_williams_hits, "Williams", verbose);

    compare_hitcounts(h_plucker_hits, "Plucker",
                      h_williams_hits, "Williams", verbose);

    compare_hitcounts(h_aila_laine_hits, "Aila",
                      h_williams_hits, "Williams", verbose);

    compare_hitcounts(h_williams_noif_hits, "Williams (no ifs)",
                      h_williams_hits, "Williams", verbose);

    return EXIT_SUCCESS;
}
