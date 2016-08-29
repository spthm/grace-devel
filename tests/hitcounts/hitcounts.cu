// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include "grace/cuda/nodes.h"
#include "grace/cuda/trace_sph.cuh"
#include "grace/cuda/generate_rays.cuh"
#include "grace/ray.h"
#include "grace/sphere.h"
#include "grace/vector.h"
#include "helper/tree.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>

typedef grace::Sphere<float> SphereType;

int main(int argc, char* argv[])
{
    size_t N = 1000000;
    size_t N_rays = 3125 * 32; // = 100,000
    int max_per_leaf = 32;
    bool save_data = false;

    if (argc > 1) {
        N = (size_t)std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = 32 * (size_t)std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        max_per_leaf = (int)std::strtol(argv[3], NULL, 10);
    }
    if (argc > 4) {
        save_data = (std::string(argv[4]) == "save") ? true : false;
    }

    if (save_data) std::cout << "Will save all data." << std::endl;
    std::cout << "Number of rays:         " << N_rays << std::endl
              << "Number of particles:    " << N << std::endl
              << "Max particles per leaf: " << max_per_leaf << std::endl
              << std::endl;

    thrust::device_vector<SphereType> d_spheres(N);
    thrust::device_vector<grace::Ray> d_rays(N_rays);
    thrust::device_vector<int> d_hit_counts(N_rays);
    grace::Tree d_tree(N, max_per_leaf);

    // Random spheres in [0, 1) are generated, with radii in [0, 0.1).
    SphereType high = SphereType(1.f, 1.f, 1.f, 0.1f);
    SphereType low = SphereType(0.f, 0.f, 0.f, 0.f);
    // Rays emitted from box centre and of sufficient length to exit the box.
    grace::Vector<3, float> origin = grace::Vector<3, float>(.5f, .5f, .5f);
    float length = 2.f;

    random_spheres_tree(low, high, N, d_spheres, d_tree);
    grace::uniform_random_rays(d_rays, origin, length);
    grace::trace_hitcounts_sph(d_rays, d_spheres, d_tree, d_hit_counts);

    int max_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(), 0,
                                  thrust::maximum<int>());
    int min_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(), N,
                                  thrust::minimum<int>());
    int total_hits = thrust::reduce(d_hit_counts.begin(), d_hit_counts.end(), 0,
                                    thrust::plus<int>());
    std::cout << "Total hits: " << total_hits << std::endl
              << "Max hits:   " << max_hits << std::endl
              << "Min hits:   " << min_hits << std::endl;

    if (save_data)
    {
        std::ofstream outfile;
        outfile.setf(std::ios::fixed, std::ios::floatfield);
        outfile.precision(8);

        thrust::host_vector<SphereType> h_spheres_xyzr = d_spheres;
        thrust::host_vector<grace::Ray> h_rays = d_rays;

        outfile.open("indata/spheredata.txt");
        for (int i=0; i<N; i++) {
            outfile << h_spheres_xyzr[i].x << " " << h_spheres_xyzr[i].y << " "
                    << h_spheres_xyzr[i].z << " " << h_spheres_xyzr[i].r
                    << std::endl;
        }
        outfile.close();

        outfile.open("indata/raydata.txt");
        for (int i=0; i<N_rays; i++) {
            outfile << h_rays[i].dx    << " " << h_rays[i].dy << " "
                    << h_rays[i].dz    << " " << h_rays[i].ox << " "
                    << h_rays[i].oy    << " " << h_rays[i].oz << " "
                    << h_rays[i].start << " " << h_rays[i].end
                    << std::endl;
        }
        outfile.close();

        thrust::host_vector<int> h_hit_counts = d_hit_counts;
        outfile.open("outdata/hitdata.txt");
        for (int i=0; i<N_rays; i++) {
            outfile << h_hit_counts[i] << std::endl;
        }
        outfile.close();
    }

    return EXIT_SUCCESS;
}
