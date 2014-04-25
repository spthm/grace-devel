#include <math.h>

#include <curand.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../kernels/gen_rays.cuh"
#include "../ray.h"

int main(void) {

    size_t N_rays = 10000;
    double inv_N_rays = 1. / N_rays;
    // Density of uniform points on unit sphere / pi.
    double prefactor = 4 * inv_N_rays;

    thrust::device_vector<grace::Ray> d_rays(N_rays);
    thrust::host_vector<grace::Ray> h_rays(N_rays);

    float ox, oy, oz;
    ox = oy = oz = 0;
    float length = 1;

    grace::uniform_random_rays(d_rays, ox, oy, oz, length);

    // h_rays = d_rays;

    // Loop through various distance scales.
    for (float t=sqrt(2); t>0; t-=0.1)
    {
        double Lt;
        unsigned int N_within = 0;
        // Loop through each particle.
        for (unsigned int i=0; i<N_rays; i++)
        {
            float dxi, dyi, dzi;
            dxi = h_rays[i].dx;
            dyi = h_rays[i].dy;
            dzi = h_rays[i].dz;

            for (unsigned int j=0; j<N_rays; j++)
            {
                if (i == j)
                    continue;

                float dxj, dyj, dzj;
                dxj = h_rays[j].dx;
                dyj = h_rays[j].dy;
                dzj = h_rays[j].dz;

                if ( (dxi-dxj)*(dxi-dxj) +
                     (dyi-dyj)*(dyi-dyj) +
                     (dzi-dzj)*(dzi-dzj) < t*t)
                {
                    N_within++;
                }
            }
        }
        // Convert to a per-paricle average.
        N_within *= inv_N_rays;
        // Calculate L(t), which is sqrt(K(t)/pi), where K(t) is
        // Ripley's K function
        Lt = sqrt(prefactor * N_within);
        std::cout << "L(" << t << ") = " << Lt << std::endl;
    }
}
