#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

// See:
// https://code.google.com/p/thrust/source/browse/examples/monte_carlo.cu
// as well as:
// http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
// and links therin.


// Thomas Wang hash.
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

class random_float_functor
{
    const unsigned int offset;
    const float scale;
    const unsigned int seed_factor;

public:
    random_float_functor() : offset(0u), seed_factor(1u), scale(1.0) {}

    explicit random_float_functor(const unsigned int offset_) :
        offset(offset_), scale(1.0), seed_factor(1u) {}

    explicit random_float_functor(const float scale_) :
        offset(0u), scale(scale_), seed_factor(1u) {}

    explicit random_float_functor(const unsigned int offset_,
                                  const float scale_) :
        offset(offset_), scale(scale_), seed_factor(1u) {}

    explicit random_float_functor(const unsigned int offset_,
                                  const unsigned int seed_factor_) :
        offset(offset_), scale(1.0), seed_factor(seed_factor_) {}

    explicit random_float_functor(const float scale_,
                                  const unsigned int seed_factor_) :
        offset(0u), scale(scale_), seed_factor(seed_factor_) {}

    explicit random_float_functor(const unsigned int offset_,
                                  const float scale_,
                                  const unsigned int seed_factor_) :
        offset(offset_), scale(scale_), seed_factor(seed_factor_) {}

    __host__ __device__ float operator() (unsigned int n)
    {
        unsigned int seed = n;
        for (int i=0; i<seed_factor; i++) {
            seed = hash(seed);
        }
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<float> u01(0,1);

        rng.discard(offset);

        return scale*u01(rng);
    }
};

#define SKIP_SPACER(file) file.get();

struct gadget_header
{
  int npart[6];
  double mass[6];
  // double time;
  // double redshift;
  // int flag_sfr;
  // int flag_feedback;
  // int npartTotal[6];
  // int flag_cooling;
  // int num_files;
  // double BoxSize;
  // double Omega0;
  // double OmegaLambda;
  // double HubbleParam;
  char fill [256 - 6*4 - 6*8];
  // char fill[256 - 6 * 4 - 6 * 8 - 2 * 8 - 2 * 4 - 6 * 4 - 2 * 4 - 4 * 8];   /* fills to 256 Bytes */
};

gadget_header read_gadget_header(std::ifstream &file) {
    gadget_header header;
    SKIP_SPACER(file)
    file.read((char*)&header.npart, sizeof(int)*6);
    file.read((char*)&header.mass, sizeof(double)*6);
    file.read((char*)&header.fill, sizeof(header.fill));
    SKIP_SPACER(file)
    return header;
}

void read_gadget_gas(std::ifstream &file,
                     thrust::host_vector<float> x,
                     thrust::host_vector<float> y,
                     thrust::host_vector<float> z,
                     thrust::host_vector<float> h)
{
    int i_dummy;
    float f_dummy;
    int N_gas, N_withmasses;

    // Skip to end of header.
    file.seekg(std::ios::beg);
    gadget_header header = read_gadget_header(file);

    // Calculate particle number counts, and read in positions block.
    N_gas = header.npart[0];
    N_withmasses = 0;
    SKIP_SPACER(file)
    for(int i=0; i<6; i++) {
        if (header.mass[i] == 0)
            N_withmasses += header.npart[i];

        for(int n=0; n<header.npart[i]; n++) {
            file.read((char*)&x[n], sizeof(float));
            file.read((char*)&y[n], sizeof(float));
            file.read((char*)&z[n], sizeof(float));
        }
    }
    SKIP_SPACER(file)

    // Velocities.
    SKIP_SPACER(file)
    for(int i=0; i<6; i++) {
        for(int n=0; n<header.npart[i]; n++) {
            file.read((char*)&f_dummy, sizeof(float));
            file.read((char*)&f_dummy, sizeof(float));
            file.read((char*)&f_dummy, sizeof(float));
        }
    }
    SKIP_SPACER(file)

    // IDs.
    SKIP_SPACER(file) {
    for(int i=0; i<6; i++)
        for(int n=0; n<header.npart[i]; n++) {
            file.read((char*)&i_dummy, sizeof(int));
        }
    }
    SKIP_SPACER(file)

    // Masses (optional).
    if (N_withmasses > 0)
        SKIP_SPACER(file)
    for(int i=0; i<6; i++) {
        if (header.mass[i] == 0) {
            for (int n=0; n<header.npart[i]; n++) {
                file.read((char*)&f_dummy, sizeof(float));
            }
        }
    }
    if (N_withmasses > 0)
        SKIP_SPACER(file)

    // Gas properties (optional).
    if (N_gas > 0)
    {
        // Internal energies.
        SKIP_SPACER(file)
        for(int n=0; n<N_gas; n++) {
            file.read((char*)&f_dummy, sizeof(float));
        }
        SKIP_SPACER(file)

        // Densities.
        SKIP_SPACER(file)
        for (int n=0; n<N_gas; n++) {
            file.read((char*)&f_dummy, sizeof(float));
        }
        SKIP_SPACER(file)

        // Smoothing lengths.
        SKIP_SPACER(file)
        for (int n=0; n<N_gas; n++) {
            file.read((char*)&h[n], sizeof(float));
        }
        SKIP_SPACER(file)
    }

    return;

}
