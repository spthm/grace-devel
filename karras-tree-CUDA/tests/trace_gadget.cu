#include <cmath>
#include <sstream>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "utils.cuh"
#include "../types.h"
#include "../nodes.h"
#include "../ray.h"
#include "../kernels/morton.cuh"
#include "../kernels/bintree_build_kernels.cuh"
#include "../kernels/bintree_trace.cuh"

int main(int argc, char* argv[])
{
    typedef grace::Vector3<float> Vector3f;

    unsigned int N_rays = 250000;
    if (argc > 1) {
        N_rays = (unsigned int) std::strtol(argv[1], NULL, 10);
    }
    float N_rays_side = floor(pow(N_rays, 0.500001));

    std::ifstream file;
    std::string fname = "Data_025";
    std::cout << "Reading in data from Gadget file " << fname << "..."
              << std::endl;

    // Read in gas data from Gadget-2 file.
    // Arrays are resized in read_gadget_gas().
    thrust::host_vector<float> h_x_centres(1);
    thrust::host_vector<float> h_y_centres(1);
    thrust::host_vector<float> h_z_centres(1);
    thrust::host_vector<float> h_radii(1);
    thrust::host_vector<float> h_masses(1);

    file.open(fname.c_str(), std::ios::binary);
    read_gadget_gas(file, h_x_centres, h_y_centres, h_z_centres,
                          h_radii, h_masses);
    file.close();
    // Masses unused.
    h_masses.clear(); h_masses.shrink_to_fit();

    unsigned int N = h_x_centres.size();
    // Factor of two is a fudge.
    unsigned int N_hits_per_ray = floor(pow(N, 0.333334));
    std::cout << "Will trace " << N_rays << " rays through " << N
              << " particles..." << std::endl;
    std::cout << std::endl;

// Device code.
{
    thrust::device_vector<float> d_x_centres = h_x_centres;
    thrust::device_vector<float> d_y_centres = h_y_centres;
    thrust::device_vector<float> d_z_centres = h_z_centres;
    thrust::device_vector<float> d_radii = h_radii;

    // Set the AABBs.
    float max_x = thrust::reduce(h_x_centres.begin(),
                                 h_x_centres.end(),
                                 -1.0f,
                                 thrust::maximum<float>());
    float max_y = thrust::reduce(h_y_centres.begin(),
                                 h_y_centres.end(),
                                 -1.0f,
                                 thrust::maximum<float>());
    float max_z = thrust::reduce(h_z_centres.begin(),
                                 h_z_centres.end(),
                                 -1.0f,
                                 thrust::maximum<float>());
    float min_x = thrust::reduce(h_x_centres.begin(),
                                 h_x_centres.end(),
                                 max_x,
                                 thrust::minimum<float>());
    float min_y = thrust::reduce(h_y_centres.begin(),
                                 h_y_centres.end(),
                                 max_y,
                                 thrust::minimum<float>());
    float min_z = thrust::reduce(h_z_centres.begin(),
                                 h_z_centres.end(),
                                 max_z,
                                 thrust::minimum<float>());
    float max_r = thrust::reduce(h_radii.begin(),
                                 h_radii.end(),
                                 -1.0f,
                                 thrust::minimum<float>());
    Vector3f bottom(min_x, min_y, min_z);
    Vector3f top(max_x, max_y, max_z);

    // Generate morton keys based on particles' positions.
    thrust::device_vector<UInteger32> d_keys(N);
    grace::morton_keys(d_x_centres, d_y_centres, d_z_centres, d_keys,
                       bottom, top);

    // Sort all particle arrays by their keys.
    thrust::device_vector<int> d_indices(N);
    thrust::device_vector<float> d_tmp(N);
    thrust::sequence(d_indices.begin(), d_indices.end());
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

    thrust::gather(d_indices.begin(),
                   d_indices.end(),
                   d_radii.begin(),
                   d_tmp.begin());
    d_radii = d_tmp;
    // Clear temporary storage.
    d_tmp.clear(); d_tmp.shrink_to_fit();
    d_indices.clear(); d_indices.shrink_to_fit();

    // Build the tree hierarchy from the keys.
    thrust::device_vector<grace::Node> d_nodes(N-1);
    thrust::device_vector<grace::Leaf> d_leaves(N);
    grace::build_nodes(d_nodes, d_leaves, d_keys);
    // Keys no longer needed.
    d_keys.clear(); d_keys.shrink_to_fit();
    grace::find_AABBs(d_nodes, d_leaves,
                      d_x_centres, d_y_centres, d_z_centres, d_radii);

    // Generate the rays, emitted from box side (X, Y, min_z-max_r) and of
    // length (max_z + max_r) - (min_z - max_r).
    float span_x = 2*max_r + max_x - min_x;
    float span_y = 2*max_r + max_y - min_y;
    float span_z = 2*max_r + max_z - min_z;
    float spacer_x = span_x / N_rays_side;
    float spacer_y = span_y / N_rays_side;
    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<UInteger32> h_keys(N_rays);
    int i, j;
    float ox, oy;
    for (i=0, ox=(min_x-max_r); i<N_rays_side; ox+=spacer_x, i++)
    {
        for (j=0, oy=(min_y-max_r); j<N_rays_side; oy+=spacer_y, j++)
        {
            // All rays point in +ve z direction.
            h_rays[i*N_rays_side +j].dx = 0.0f;
            h_rays[i*N_rays_side +j].dy = 0.0f;
            h_rays[i*N_rays_side +j].dz = 1.0f;

            h_rays[i*N_rays_side +j].ox = ox;
            h_rays[i*N_rays_side +j].oy = oy;
            h_rays[i*N_rays_side +j].oz = min_z - max_r;

            h_rays[i*N_rays_side +j].length = span_z;
            h_rays[i*N_rays_side +j].dclass = 7;
        }

        // Since all rays are PPP, base key on origin instead.
        // morton_key(float, float, float) requires floats in (0, 1).
        h_keys[i] = grace::morton_key((ox-(min_x-max_r))/span_x,
                                      (oy-(min_y-max_r))/span_y,
                                      0.0f);
    }

    // Sort rays by Morton key.
    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());
    thrust::device_vector<grace::Ray> d_rays = h_rays;
    thrust::device_vector<int> d_hits(N_hits_per_ray*N_rays);
    thrust::device_vector<int> d_hit_count(N_rays);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    grace::gpu::trace<<<28, TRACE_THREADS_PER_BLOCK>>>
                     (thrust::raw_pointer_cast(d_rays.data()),
                      d_rays.size(),
                      N_hits_per_ray,
                      thrust::raw_pointer_cast(d_hits.data()),
                      thrust::raw_pointer_cast(d_hit_count.data()),
                      thrust::raw_pointer_cast(d_nodes.data()),
                      thrust::raw_pointer_cast(d_leaves.data()),
                      thrust::raw_pointer_cast(d_x_centres.data()),
                      thrust::raw_pointer_cast(d_y_centres.data()),
                      thrust::raw_pointer_cast(d_z_centres.data()),
                      thrust::raw_pointer_cast(d_radii.data()));
    CUDA_HANDLE_ERR( cudaPeekAtLastError() );
    CUDA_HANDLE_ERR( cudaDeviceSynchronize() );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    int max_hits = thrust::reduce(d_hit_count.begin(), d_hit_count.end(),
                                  0, thrust::maximum<int>());
    int min_hits = thrust::reduce(d_hit_count.begin(), d_hit_count.end(),
                                  N, thrust::minimum<int>());
    float mean_hits = thrust::reduce(d_hit_count.begin(), d_hit_count.end(),
                                     0, thrust::plus<int>()) / float(N_rays);
    std::cout << "Time for tracing kernel: " << elapsed << " ms" << std::endl;
    std::cout << std::endl;
    std::cout << "Number of rays:       " << N_rays << std::endl;
    std::cout << "Number of particles:  " << N << std::endl;
    std::cout << "Expected hit count:   " << N_hits_per_ray << std::endl;
    std::cout << "Mean hits:            " << mean_hits << std::endl;
    std::cout << "Max hits:             " << max_hits << std::endl;
    std::cout << "Min hits:             " << min_hits << std::endl;

    // Sort ray hit data such that increasing the index moves us along x first,
    // then y.
    thrust::host_vector<float> h_pos_keys(N_rays);
    for (int i=0; i<N_rays; i++) {
        h_pos_keys[i] = h_rays[i].ox + (2*span_x)*h_rays[i].oy;
    }
    thrust::host_vector<float> h_hit_count = d_hit_count;
    thrust::sort_by_key(h_pos_keys.begin(), h_pos_keys.end(),
                        h_hit_count.begin());
    for (int i=0; i<N_rays; i++) {
        h_pos_keys[i] = h_rays[i].ox + (2*span_x)*h_rays[i].oy;
    }
    thrust::sort_by_key(h_pos_keys.begin(), h_pos_keys.end(),
                        h_rays.begin());

    FILE *f;
    unsigned char *img = NULL;
    int w=N_rays_side;
    int h=N_rays_side;
    int filesize = 54 + 3*w*h;  //w is your image width, h is image height, both int
    if( img )
        free( img );
    img = (unsigned char *)malloc(3*w*h);
    memset(img,0,sizeof(img));

    int r, g, b, x, y;
    for(int i=0; i<w; i++)
    {
        for(int j=0; j<h; j++)
    {
        x=i; y=(h-1)-j;
        r = g = b = (int) (h_hit_count[i+w*j]*255.0/max_hits);
        if (r > 255) r=255;
        if (g > 255) g=255;
        if (b > 255) b=255;
        img[(x+y*w)*3+2] = (unsigned char)(r);
        img[(x+y*w)*3+1] = (unsigned char)(g);
        img[(x+y*w)*3+0] = (unsigned char)(b);
        //std::cout << "ox, oy: " << h_rays[i+w*j].ox << ", " << h_rays[i+w*j].oy
        //          << std::endl;
    }
    }

    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(       w    );
    bmpinfoheader[ 5] = (unsigned char)(       w>> 8);
    bmpinfoheader[ 6] = (unsigned char)(       w>>16);
    bmpinfoheader[ 7] = (unsigned char)(       w>>24);
    bmpinfoheader[ 8] = (unsigned char)(       h    );
    bmpinfoheader[ 9] = (unsigned char)(       h>> 8);
    bmpinfoheader[10] = (unsigned char)(       h>>16);
    bmpinfoheader[11] = (unsigned char)(       h>>24);

    f = fopen("img.bmp","wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    for(i=0; i<h; i++)
    {
        fwrite(img+(w*(h-i-1)*3),3,w,f);
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }
    fclose(f);
} // Device code.  Call all thrust destructors etc. before cudaDeviceReset().

    // Exit cleanly to ensure full profiler trace.
    cudaDeviceReset();
    return 0;
}
