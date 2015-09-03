#include <cmath>
#include <sstream>
#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "../nodes.h"
#include "../ray.h"
#include "../utils.cuh"
#include "../kernels/bintree_build.cuh"
#include "../kernels/bintree_trace.cuh"
#include "../kernels/morton.cuh"
#include "../kernels/sort.cuh"

int main(int argc, char* argv[]) {

    /* Initialize run parameters. */

    unsigned int N_rays = 512*512;
    unsigned int max_per_leaf = 32;

    if (argc > 1)
        N_rays = 32 * (unsigned int) std::strtol(argv[1], NULL, 10);
    if (argc > 2)
        max_per_leaf = (unsigned int) std::strtol(argv[2], NULL, 10);

    unsigned int N_rays_side = floor(pow(N_rays, 0.500001));


    /* Read in Gadget file. */

    std::ifstream file;
    std::string fname = "Data_025";
    std::cout << "Reading in data from Gadget file " << fname << "..."
              << std::endl;

    // Arrays are resized in read_gadget_gas().
    thrust::host_vector<float4> h_spheres_xyzr(1);
    thrust::host_vector<unsigned int> h_gadget_IDs(1);
    thrust::host_vector<float> h_masses(1);
    thrust::host_vector<float> h_rho(1);

    file.open(fname.c_str(), std::ios::binary);
    grace::read_gadget_gas(file, h_spheres_xyzr,
                                 h_gadget_IDs,
                                 h_masses,
                                 h_rho);
    file.close();

    size_t N = h_spheres_xyzr.size();
    std::cout << "Will trace " << N_rays << " rays through " << N
              << " particles, with up to " << max_per_leaf << " particle(s)"
              << std::endl
              << "per leaf..." << std::endl;
    std::cout << std::endl;

    // Gadget IDs, masses and densities unused.
    h_gadget_IDs.clear(); h_gadget_IDs.shrink_to_fit();
    h_masses.clear(); h_masses.shrink_to_fit();
    h_rho.clear(); h_rho.shrink_to_fit();


{ // Device code.


    /* Build the tree. */

    thrust::device_vector<float4> d_spheres_xyzr = h_spheres_xyzr;

    // Calculate limits here explicity since we need them later (i.e. do not
    // get morton_keys() to do it for us).
    float min_x, max_x;
    grace::min_max_x(&min_x, &max_x, d_spheres_xyzr);

    float min_y, max_y;
    grace::min_max_y(&min_y, &max_y, d_spheres_xyzr);

    float min_z, max_z;
    grace::min_max_z(&min_z, &max_z, d_spheres_xyzr);

    float min_r, max_r;
    grace::min_max_w(&min_r, &max_r, d_spheres_xyzr);

    float3 top = make_float3(max_x, max_y, max_z);
    float3 bot = make_float3(min_x, min_y, min_z);

    thrust::device_vector<unsigned int> d_keys(N);

    grace::morton_keys(d_keys, d_spheres_xyzr, top, bot);
    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_spheres_xyzr.begin());

    grace::Tree d_tree(N, max_per_leaf);
    thrust::device_vector<float> d_deltas(N + 1);

    grace::compute_deltas(d_spheres_xyzr, d_deltas);
    grace::build_tree(d_tree, d_spheres_xyzr, d_deltas, d_spheres_xyzr);

    // Keys and deltas no longer needed.
    d_keys.clear(); d_keys.shrink_to_fit();
    d_deltas.clear(); d_deltas.shrink_to_fit();


    /* Generate the rays, all emitted in +z direction from a box side. */

    // Rays emitted from box side (x, y, min_z - max_r) and of length
    // (max_z + max_r) - (min_z - max_r).  For simplicity, the ray (ox, oy)
    // limits are determined only by the particle min(x, y) / max(x, y) limits
    // and smoothing lengths are ignored.  This ensures that rays at the edge
    // will hit something!
    float span_x = max_x - min_x;
    float span_y = max_y - min_y;
    float span_z = 2*max_r + max_z - min_z;
    float spacer_x = span_x / (N_rays_side-1);
    float spacer_y = span_y / (N_rays_side-1);

    thrust::host_vector<grace::Ray> h_rays(N_rays);
    thrust::host_vector<unsigned int> h_keys(N_rays);

    int i, j;
    float ox, oy;
    for (i=0, ox=min_x; i<N_rays_side; ox+=spacer_x, i++)
    {
        for (j=0, oy=min_y; j<N_rays_side; oy+=spacer_y, j++)
        {
            h_rays[i*N_rays_side + j].dx = 0.0f;
            h_rays[i*N_rays_side + j].dy = 0.0f;
            h_rays[i*N_rays_side + j].dz = 1.0f;

            h_rays[i*N_rays_side + j].ox = ox;
            h_rays[i*N_rays_side + j].oy = oy;
            h_rays[i*N_rays_side + j].oz = min_z - max_r;

            h_rays[i*N_rays_side + j].length = span_z;

            // Since all rays are PPP, base key on origin instead.
            // Floats must be in (0, 1) for morton_key().
            h_keys[i*N_rays_side + j] = grace::morton_key((ox-min_x)/span_x,
                                                          (oy-min_y)/span_y,
                                                          0.0f);
        }
    }


    /* Trace and accumulate density through the similation data. */

    thrust::sort_by_key(h_keys.begin(), h_keys.end(), h_rays.begin());
    thrust::device_vector<grace::Ray> d_rays = h_rays;

    thrust::device_vector<float> d_traced_integrals(N_rays);
    grace::KernelIntegrals<float> lookup;
    thrust::device_vector<float> d_b_integrals(&lookup.table[0],
                                               &lookup.table[50]);

    grace::trace_cumulative(d_rays,
                            d_traced_integrals,
                            d_tree,
                            d_spheres_xyzr);

    float max_integral = thrust::reduce(d_traced_integrals.begin(),
                                        d_traced_integrals.end(),
                                        0.0f, thrust::maximum<float>());
    float min_integral = thrust::reduce(d_traced_integrals.begin(),
                                        d_traced_integrals.end(),
                                        1E20, thrust::minimum<float>());
    float mean_integral = thrust::reduce(d_traced_integrals.begin(),
                                         d_traced_integrals.end(),
                                         0.0f, thrust::plus<float>());
    mean_integral /= static_cast<float>(d_traced_integrals.size());

    std::cout << "Number of rays:         " << N_rays << std::endl;
    std::cout << "Number of particles:    " << N << std::endl;
    std::cout << "Mean output             " << mean_integral << std::endl;
    std::cout << "Max output:             " << max_integral << std::endl;
    std::cout << "Min output:             " << min_integral << std::endl;
    std::cout << std::endl;


    /* Generate an image of the projected density of the simulation volume. */

    // Sort ray hit and ray data such that increasing the index moves us along
    // x first, then y.
    thrust::host_vector<float> h_pos_keys(N_rays);
    for (int i=0; i<N_rays; i++) {
        h_pos_keys[i] = h_rays[i].ox + (2*span_x)*h_rays[i].oy;
    }
    thrust::host_vector<float> h_traced_integrals = d_traced_integrals;
    thrust::host_vector<int> h_indices(N_rays);

    grace::sort_by_key(h_pos_keys, h_traced_integrals, h_rays);

    // Increase the dynamic range.
    for (int i=0; i<N_rays; i++) {
        h_traced_integrals[i] = log10(h_traced_integrals[i]);
    }
    min_integral = log10(min_integral);
    max_integral = log10(max_integral);

    // See http://stackoverflow.com/questions/2654480
    FILE *f;
    unsigned char *img = NULL;
    int w = N_rays_side;
    int h = N_rays_side;
    int filesize = 54 + 3*w*h;

    img = (unsigned char *)malloc(3*w*h);
    memset(img,0,sizeof(img));

    int r, g, b, x, y;
    float r_max = 150.0f;
    float g_max = 210.0f;
    float b_max = 255.0f;
    for(int i=0; i<w; i++)
    {
        for(int j=0; j<h; j++)
    {
        x = i; y = (h-1) - j;
        r = (int)( (h_traced_integrals[i+w*j] - min_integral) * r_max/(max_integral - min_integral) );
        g = (int)( (h_traced_integrals[i+w*j] - min_integral) * g_max/(max_integral - min_integral) );
        b = (int)( (h_traced_integrals[i+w*j] - min_integral) * b_max/(max_integral - min_integral) );
        if (r > r_max) r = r_max;
        if (g > g_max) g = g_max;
        if (b > b_max) b = b_max;
        img[(x+y*w)*3+2] = (unsigned char)(r);
        img[(x+y*w)*3+1] = (unsigned char)(g);
        img[(x+y*w)*3+0] = (unsigned char)(b);
    }
    }

    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};

    bmpfileheader[2] = (unsigned char)(filesize);
    bmpfileheader[3] = (unsigned char)(filesize >> 8);
    bmpfileheader[4] = (unsigned char)(filesize >> 16);
    bmpfileheader[5] = (unsigned char)(filesize >> 24);

    bmpinfoheader[4] =  (unsigned char)(w);
    bmpinfoheader[5] =  (unsigned char)(w >> 8);
    bmpinfoheader[6] =  (unsigned char)(w >> 16);
    bmpinfoheader[7] =  (unsigned char)(w >> 24);
    bmpinfoheader[8] =  (unsigned char)(h);
    bmpinfoheader[9] =  (unsigned char)(h >> 8);
    bmpinfoheader[10] = (unsigned char)(h >> 16);
    bmpinfoheader[11] = (unsigned char)(h >> 24);

    f = fopen("density.bmp", "wb");
    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);
    for(i=0; i<h; i++)
    {
        fwrite(img+(w*(h-i-1)*3),3,w,f);
        fwrite(bmppad,1,(4-(w*3)%4)%4,f);
    }
    fclose(f);
} // End device code.

    // Exit cleanly to ensure a full profiler trace.
    cudaDeviceReset();
    return 0;
}
