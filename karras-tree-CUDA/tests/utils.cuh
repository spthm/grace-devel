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
