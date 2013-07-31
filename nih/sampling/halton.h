#pragma once

#include <nih/linalg/vector.h>
#ifdef __CUDACC__
#include <cuda_runtime_api.h>
#endif

namespace nih {

namespace halton {
extern uint32 s_halton_bases[2000];
extern float  s_halton_sigma[];
} // namespace halton

inline NIH_HOST_DEVICE uint32 halton_bases(const int32 b, const host_space tag)   { return halton::s_halton_bases[b]; }
inline NIH_HOST_DEVICE float  halton_sigma(const int32 b, const host_space tag)   { return halton::s_halton_sigma[b]; }
inline NIH_HOST_DEVICE float  halton_sigma(const int32 b, const int32 bj, const host_space tag)     { return halton::s_halton_sigma[b] / float(bj); }

#ifdef __CUDACC__

texture<float, 1>   tex_halton_sigma;
__constant__ uint32 c_halton_bases[2000];

#if __CUDA_ARCH__

inline __host__ __device__ uint32 halton_bases(const int32 b, const device_space tag) { return c_halton_bases[b]; }
inline __host__ __device__ float  halton_sigma(const int32 b, const device_space tag) { return tex1Dfetch( tex_halton_sigma, b ); }
inline __host__ __device__ float  halton_sigma(const int32 b, const int32 bj, const device_space tag) { return __fdividef( tex1Dfetch( tex_halton_sigma, b ), float(bj) ); }

#else

inline __host__ __device__ uint32 halton_bases(const int32 b, const device_space tag) { return 0; }
inline __host__ __device__ float  halton_sigma(const int32 b, const device_space tag) { return 0.0f; }
inline __host__ __device__ float  halton_sigma(const int32 b, const int32 bj, const device_space tag) { return 0.0f; }

#endif

#endif

///
/// A Halton QMC sampler
///
template <typename space_tag = host_space>
class Halton_sampler
{
public:
	inline NIH_HOST_DEVICE Halton_sampler ();
	  // Constructs an empty sampling context.
	  // Child contexts will inherit the randomization seed.

	inline NIH_HOST_DEVICE ~Halton_sampler ();
	  // Destructor

	inline NIH_HOST_DEVICE Halton_sampler split (const uint32 dim, const uint32 n, const uint32 start = 0);
	  // Dependent splitting function, creating a child sampling
	  // context with 'dim' dimensions and 'n' samples.

    inline NIH_HOST_DEVICE bool  next_sample(float* u);
    inline NIH_HOST_DEVICE float sample(const int32 i);

    inline NIH_HOST_DEVICE bool  advance() { return ++dLI < dLN; }
	  // Next sample from the sampling loop. Returns 0 if the
	  // loop is already finished, 1 otherwise.

	inline static void init(void);
	  // Initialization function

	inline NIH_HOST_DEVICE static float phi (uint32 n, uint32 b);
	  // Radical inverse function

	static const uint32  sMAXDIM = 200;
    static const uint32  sBASESCOUNT = 9974;

private:
	uint32 dGI; // Global instance number
	uint32 dGD; // Global integral dimension
	uint32 dLI; // Local instance number
	uint32 dLD; // Local integral dimension
	uint32 dLN; // Local number of samples
};

#ifdef __CUDACC__
struct Halton_sampler_binder
{
    Halton_sampler_binder()
    {
        // copy bases & sigma to constant memory
        cudaMemcpyToSymbol( c_halton_bases, halton::s_halton_bases, sizeof(uint32)*2000 );

        const uint32 bases_count = Halton_sampler<device_space>::sBASESCOUNT;

        cudaMalloc( (void**)&g_sigma, sizeof(float)*bases_count*bases_count );
        //cudaMalloc( (void**)&g_bases, sizeof(uint32)*2000 );
        if (g_sigma == NULL /*|| g_bases == NULL*/)
        {
            fprintf(stderr, "Halton_sampler_binder(): allocation failed\n");
            return;
        }
        cudaMemcpy( g_sigma, halton::s_halton_sigma, sizeof(float)*bases_count*bases_count, cudaMemcpyHostToDevice );
        //cudaMemcpy( g_bases, halton::s_halton_bases, sizeof(int32)*2000, cudaMemcpyHostToDevice );

        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();
        cudaBindTexture( 0, &tex_halton_sigma, g_sigma, &channel_desc, bases_count*bases_count );
    }
    ~Halton_sampler_binder()
    {
        //cudaUnbindTexture( &tex_halton_sigma );
        cudaFree( g_sigma );
        cudaFree( g_bases );
    }

    float* g_sigma;
    int32* g_bases;
};
#endif

// ============================================================================ //
//                             IMPLEMENTATION                                   //
// ============================================================================ //

template <typename space_tag>
inline Halton_sampler<space_tag>::Halton_sampler () :
	dGI ( 0 ), dGD( 0 ), dLI( 0 ), dLD( 0 ), dLN( 0 )
{
}
template <typename space_tag>
inline Halton_sampler<space_tag>::~Halton_sampler ()
{
}
template <typename space_tag>
inline Halton_sampler<space_tag> Halton_sampler<space_tag>::split (const uint32 dim, const uint32 n, const uint32 start)
{
	Halton_sampler ctxt;
	ctxt.dGI = dGI + dLI; // Decorrelate dependent samples
	ctxt.dGD = dGD + dim; // Compute global dimension
	ctxt.dLI = start;
	ctxt.dLD = dim;
	ctxt.dLN = n + start;
	return ctxt;
}
template <typename space_tag>
inline bool Halton_sampler<space_tag>::next_sample(float* u)
{
	if (dLI == dLN) // Check if loop is finished
		return false;

    for (uint32 j = 0; j < dLD; ++j)
        u[j] = sample( j );

    return advance();
}
template <typename space_tag>
inline float Halton_sampler<space_tag>::sample (const int32 j)
{
	// Build sample point
    const float r = dGI ? 
        fmodf(
            phi( dGI, dGD - dLD + j ) +
            phi( dLI, j ), 1.0f ) :
            phi( dLI, j );
    return r;
}

template <typename space_tag>
inline float Halton_sampler<space_tag>::phi (uint32 n, uint32 bindex)
{
	float  result = 0.0;
	uint32  remainder;
	uint32  m, bj = 1, b = halton_bases( bindex, space_tag() );

	do
	{
		bj *= b;
		m   = n;
		n  /= b;

		remainder = m - n * b;

		result += halton_sigma( b*sBASESCOUNT + remainder, bj, space_tag() );
	} while (n > 0);

	return result;
}

#define SIGMA(b,i) halton::s_halton_sigma[ ((b) * sBASESCOUNT + (i)) ]

template <typename space_tag>
inline void Halton_sampler<space_tag>::init (void)
{
	// Build Faure permutation tables
	SIGMA(2,0) = 0;
	SIGMA(2,1) = 1;
	for (uint32 b = 3; b < sBASESCOUNT; b++)
	{
		uint32 i;

		if (b % 2 == 0)
		{
			uint32 bByHalf = b / 2;

			for (i = 0; i < bByHalf; i++)
				SIGMA( b, i ) = 2 * SIGMA( bByHalf, i );

			for (i = 0; i < bByHalf; i++)
				SIGMA( b, bByHalf + i ) = 2 * SIGMA( bByHalf, i ) + 1;
		}
		else
		{
			uint32 bMinusOneByHalf = (b - 1) / 2;

			for (i = 0; i < bMinusOneByHalf; i++)
			{
				float s = SIGMA( b-1, i );
				SIGMA( b, i ) = s < bMinusOneByHalf ? s : s+1;
			}

			for (i = bMinusOneByHalf; i < b-1; i++)
			{
				float s = SIGMA( b-1, i );
				SIGMA( b, i+1 ) = s < bMinusOneByHalf ? s : s+1;
			}

			SIGMA( b, bMinusOneByHalf ) = (float)bMinusOneByHalf;
		}
	}
}

#undef SIGMA

} // namespace nih
