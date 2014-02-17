#include <fstream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

#define CUDA_HANDLE_ERR(code) { cudaErrorCheck((code), __FILE__, __LINE__); }

inline void cudaErrorCheck(cudaError_t code, char *file, int line, bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"CUDA Error!\nCode: %s\nFile: %s @ line %d\n", cudaGetErrorString(code), file, line);

    if (abort)
      exit(code);
  }
}

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
    thrust::uniform_real_distribution<float> uniform;
    const unsigned int seed_factor;

public:
    random_float_functor() : offset(0u), seed_factor(1u),
                             uniform(0.0f, 1.0f) {}

    explicit random_float_functor(const unsigned int offset_) :
        offset(offset_), uniform(0.0f, 1.0f), seed_factor(1u) {}

    explicit random_float_functor(const float low_,
                                  const float high_) :
        offset(0u), uniform(low_, high_), seed_factor(1u) {}

    explicit random_float_functor(const unsigned int offset_,
                                  const float low_,
                                  const float high_) :
        offset(offset_), uniform(low_, high_), seed_factor(1u) {}

    explicit random_float_functor(const unsigned int offset_,
                                  const unsigned int seed_factor_) :
        offset(offset_), uniform(0.0f, 1.0f), seed_factor(seed_factor_) {}

    explicit random_float_functor(const float low_,
                                  const float high_,
                                  const unsigned int seed_factor_) :
        offset(0u), uniform(low_, high_), seed_factor(seed_factor_) {}

    explicit random_float_functor(const unsigned int offset_,
                                  const float low_,
                                  const float high_,
                                  const unsigned int seed_factor_) :
        offset(offset_), uniform(low_, high_), seed_factor(seed_factor_) {}

    __host__ __device__ float operator() (unsigned int n)
    {
        unsigned int seed = n;
        for (unsigned int i=0; i<seed_factor; i++) {
            seed = hash(seed);
        }
        thrust::default_random_engine rng(seed);

        rng.discard(offset);

        return uniform(rng);
    }
};

class random_float4_functor
{
    float4 xyzw;
    unsigned int seed;
    thrust::uniform_real_distribution<float> uniform;
    const float w_scale;
    const unsigned int seed_factor;

public:
    random_float4_functor() : uniform(0.0f, 1.0f), w_scale(1.0f),
                              seed_factor(1u) {}

    explicit random_float4_functor(const float low_,
                                  const float high_) :
        uniform(low_, high_), w_scale(1.0f), seed_factor(1u) {}

    explicit random_float4_functor(const float low_,
                                  const float high_,
                                  const float w_scale_) :
        uniform(low_, high_), w_scale(w_scale_), seed_factor(1u) {}

    explicit random_float4_functor(const float low_,
                                  const float high_,
                                  const unsigned int seed_factor_) :
        uniform(low_, high_), w_scale(1.0f), seed_factor(seed_factor_) {}

    explicit random_float4_functor(const float w_scale_,
                                  const unsigned int seed_factor_) :
        uniform(0.0f, 1.0f), w_scale(w_scale_), seed_factor(seed_factor_) {}

    explicit random_float4_functor(const float low_,
                                  const float high_,
                                  const float w_scale_,
                                  const unsigned int seed_factor_) :
        uniform(low_, high_), w_scale(w_scale_), seed_factor(seed_factor_) {}

    __host__ __device__ float4 operator() (unsigned int n)
    {
        seed = n;
        for (unsigned int i=0; i<seed_factor; i++) {
            seed = hash(seed);
        }
        thrust::default_random_engine rng(seed);

        xyzw.x = uniform(rng);
        xyzw.y = uniform(rng);
        xyzw.z = uniform(rng);
        xyzw.w = w_scale*uniform(rng);

        return xyzw;
    }
};

inline void skip_spacer(std::ifstream& file) {
    int dummy;
    file.read((char*)&dummy, sizeof(int));
}

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

gadget_header read_gadget_header(std::ifstream& file) {
    gadget_header header;
    skip_spacer(file);
    file.read((char*)&header.npart, sizeof(int)*6);
    file.read((char*)&header.mass, sizeof(double)*6);
    file.read((char*)&header.fill, sizeof(header.fill));
    skip_spacer(file);
    return header;
}

void read_gadget_gas(std::ifstream& file,
                     thrust::host_vector<float>& x,
                     thrust::host_vector<float>& y,
                     thrust::host_vector<float>& z,
                     thrust::host_vector<float>& h,
                     thrust::host_vector<float>& m,
                     thrust::host_vector<float>& rho)
{
    int i_dummy;
    float f_dummy;
    int N_gas, N_withmasses;

    // Skip to end of header.
    file.seekg(std::ios::beg);
    gadget_header header = read_gadget_header(file);

    /* ------ Gas particles have index 0 ------ */

    // Calculate particle number counts, and read in positions block.
    N_gas = header.npart[0];
    x.resize(N_gas); y.resize(N_gas); z.resize(N_gas);
    h.resize(N_gas); m.resize(N_gas); rho.resize(N_gas);

    N_withmasses = 0;
    skip_spacer(file);
    for(int i=0; i<6; i++) {
        if (header.mass[i] == 0)
            N_withmasses += header.npart[i];

        for(int n=0; n<header.npart[i]; n++) {
            // Save gas particle data only.
            if (i == 0) {
                file.read((char*)&x[n], sizeof(float));
                file.read((char*)&y[n], sizeof(float));
                file.read((char*)&z[n], sizeof(float));
            }
            else {
                file.read((char*)&f_dummy, sizeof(float));
                file.read((char*)&f_dummy, sizeof(float));
                file.read((char*)&f_dummy, sizeof(float));
            }
        }
    }
    skip_spacer(file);

    // Velocities.
    skip_spacer(file);
    for(int i=0; i<6; i++) {
        for(int n=0; n<header.npart[i]; n++) {
            file.read((char*)&f_dummy, sizeof(float));
            file.read((char*)&f_dummy, sizeof(float));
            file.read((char*)&f_dummy, sizeof(float));
        }
    }
    skip_spacer(file);

    // IDs.
    skip_spacer(file); {
    for(int i=0; i<6; i++)
        for(int n=0; n<header.npart[i]; n++) {
            file.read((char*)&i_dummy, sizeof(int));
        }
    }
    skip_spacer(file);

    // Masses (optional).  Spacers only exist if the block exists.
    // Otherwise, all particles of a given type have equal mass, saved in the
    // header.
    if (N_withmasses > 0)
        skip_spacer(file);
    else
        thrust::fill(m.begin(), m.end(), header.mass[0]);
    for(int i=0; i<6; i++) {
        if (header.mass[i] == 0) {
            for (int n=0; n<header.npart[i]; n++) {
                file.read((char*)&f_dummy, sizeof(float));
            }
        }
    }
    if (N_withmasses > 0)
        skip_spacer(file);

    // Gas properties (optional).
    if (N_gas > 0)
    {
        // Internal energies.
        skip_spacer(file);
        for(int n=0; n<N_gas; n++) {
            file.read((char*)&f_dummy, sizeof(float));
        }
        skip_spacer(file);

        // Densities.
        skip_spacer(file);
        for (int n=0; n<N_gas; n++) {
            file.read((char*)&rho[n], sizeof(float));
        }
        skip_spacer(file);

        // Smoothing lengths.
        skip_spacer(file);
        for (int n=0; n<N_gas; n++) {
            file.read((char*)&h[n], sizeof(float));
        }
        skip_spacer(file);
    }

    return;

}
