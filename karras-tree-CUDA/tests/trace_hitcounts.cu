// Due to a bug in thrust, this must appear before thrust/sort.h
// The simplest solution is to put it here, despite already being included in
// all of the includes which require it.
// See http://stackoverflow.com/questions/23352122
#include <curand_kernel.h>

#include <cmath>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "../nodes.h"
#include "../ray.h"
#include "../utils.cuh"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/bintree_trace.cuh"
#include "../kernels/gen_rays.cuh"

int main(int argc, char* argv[]) {

    std::ofstream outfile;

    outfile.setf(std::ios::fixed, std::ios::floatfield);
    outfile.precision(9);

    /* Initialize run parameters. */

    unsigned int N = 1000000;
    unsigned int N_rays = 3125*32; // = 100,000
    unsigned int max_per_leaf = 32;
    bool save_data = false;

    if (argc > 1) {
        N = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    if (argc > 2) {
        N_rays = 32 * (unsigned int) std::strtol(argv[2], NULL, 10);
    }
    if (argc > 3) {
        max_per_leaf = (unsigned int) std::strtol(argv[3], NULL, 10);
    }
    if (argc > 4) {
        if (strcmp("save", argv[4]) == 0)
            save_data = true;
    }

    std::cout << "Generating " << N << " random points and " << N_rays
              << " random rays, with up to " << max_per_leaf << " point(s) per"
              << std::endl
              << "leaf." << std::endl;
    if (save_data)
        std::cout << "Will save all data." << std::endl;
    std::cout << std::endl;

{ // Device code.

    /* Generate N random points as floats in [0,1) and radii in [0,0.1). */

    thrust::device_vector<float4> d_spheres_xyzr(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_spheres_xyzr.begin(),
                      grace::random_float4_functor(0.1f) );


    /* Build the tree from the random data. */

    thrust::device_vector<unsigned int> d_keys(N);
    float3 top = make_float3(1.f, 1.f, 1.f);
    float3 bot = make_float3(0.f, 0.f, 0.f);

    grace::morton_keys(d_keys, d_spheres_xyzr, top, bot);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres_xyzr.begin());

    grace::Tree d_tree(N);

    grace::build_tree(d_tree, d_keys, max_per_leaf);
    grace::compact_tree(d_tree);
    grace::find_AABBs(d_tree, d_spheres_xyzr);

    // Keys no longer needed.
    d_keys.clear();
    d_keys.shrink_to_fit();


    /* Generate the rays (emitted from box centre and of length 2). */

    float ox, oy, oz, length;
    ox = oy = oz = 0.5f;
    length = 2;

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    grace::uniform_random_rays(d_rays, ox, oy, oz, length);


    /* Trace for per-ray hit counts. */

    thrust::device_vector<int> d_hit_counts(N_rays);
    grace::trace_hitcounts(d_rays, d_hit_counts,
                           d_tree, d_spheres_xyzr,
                           max_per_leaf);


    /* Output simple hit-count statistics. */

    unsigned int max_hits = thrust::reduce(d_hit_counts.begin(),
                                           d_hit_counts.end(),
                                           0, thrust::maximum<unsigned int>());
    unsigned int min_hits = thrust::reduce(d_hit_counts.begin(),
                                           d_hit_counts.end(),
                                           N, thrust::minimum<unsigned int>());
    unsigned int total_hits = thrust::reduce(d_hit_counts.begin(),
                                             d_hit_counts.end(),
                                             0, thrust::plus<unsigned int>());

    std::cout << "Number of rays:       " << N_rays << std::endl;
    std::cout << "Number of particles:  " << N << std::endl;
    std::cout << "Total hits:           " << total_hits << std::endl;
    std::cout << "Max hits:             " << max_hits << std::endl;
    std::cout << "Min hits:             " << min_hits << std::endl;


    /* Save sphere, ray and hit count data. */

    if (save_data)
    {
        thrust::host_vector<float4> h_spheres_xyzr = d_spheres_xyzr;
        thrust::host_vector<grace::Ray> h_rays = d_rays;

        outfile.open("indata/spheredata.txt");
        for (int i=0; i<N; i++) {
            outfile << h_spheres_xyzr[i].x << " " << h_spheres_xyzr[i].y << " "
                    << h_spheres_xyzr[i].z << " " << h_spheres_xyzr[i].w
                    << std::endl;
        }
        outfile.close();

        outfile.open("indata/raydata.txt");
        for (int i=0; i<N_rays; i++) {
            outfile << h_rays[i].dx << " " << h_rays[i].dy << " "
                    << h_rays[i].dz << " " << h_rays[i].ox << " "
                    << h_rays[i].oy << " " << h_rays[i].oz << " "
                    << h_rays[i].length << std::endl;
        }
        outfile.close();

        thrust::host_vector<int> h_hit_counts = d_hit_counts;
        outfile.open("outdata/hitdata.txt");
        for (int i=0; i<N_rays; i++) {
            outfile << h_hit_counts[i] << std::endl;
        }
        outfile.close();
    }

} // End device code.

    // Exit cleanly to ensure a full profiler trace.
    cudaDeviceReset();
    return 0;
}
