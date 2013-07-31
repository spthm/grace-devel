// tu_random.cpp	-- Thatcher Ulrich 2003

// This source code has been donated to the Public Domain.  Do
// whatever you want with it.

// Pseudorandom number generator.

#include <stdafx.h>

#include <mwcrand/mwcrand.h>


namespace lts {
namespace mwcrand
{

	// PRNG code adapted from the complimentary-multiply-with-carry
	// code in the article: George Marsaglia, "Seeds for Random Number
	// Generators", Communications of the ACM, May 2003, Vol 46 No 5,
	// pp90-93.
	//
	// The article says:
	//
	// "Any one of the choices for seed table size and multiplier will
	// provide a RNG that has passed extensive tests of randomness,
	// particularly those in [3], yet is simple and fast --
	// approximately 30 million random 32-bit integers per second on a
	// 850MHz PC.  The period is a*b^n, where a is the multiplier, n
	// the size of the seed table and b=2^32-1.  (a is chosen so that
	// b is a primitive root of the prime a*b^n + 1.)"
	//
	// [3] Marsaglia, G., Zaman, A., and Tsang, W.  Toward a universal
	// random number generator.  _Statistics and Probability Letters
	// 8_ (1990), 35-39.

//	const uint64	a = 18782;		// for SEED_COUNT=4096, period approx 2^131104 (from Marsaglia usenet post 2003-05-13)
//	const uint64	a = 123471786;	// for SEED_COUNT=1024, period approx 2^32794
//	const uint64	a = 123554632;	// for SEED_COUNT=512, period approx 2^16410
//	const uint64	a = 8001634;	// for SEED_COUNT=256, period approx 2^8182
//	const uint64	a = 8007626;	// for SEED_COUNT=128, period approx 2^4118
//	const uint64	a = 647535442;	// for SEED_COUNT=64, period approx 2^2077
	const uint64	a = 547416522;	// for SEED_COUNT=32, period approx 2^1053
//	const uint64	a = 487198574;	// for SEED_COUNT=16, period approx  2^540
//	const uint64	a = 716514398;	// for SEED_COUNT=8, period approx 2^285


	generator::generator()
	{
		seed_random(987654321);
	}


	void	generator::seed_random(uint32 seed)
	{
		if (seed == 0) {
			// 0 is a terrible seed (probably the only bad
			// choice), substitute something else:
			seed = 12345;
		}

		// Simple pseudo-random to reseed the seeds.
		// Suggested by the above article.
		uint32	j = seed;
		for (int i = 0; i < SEED_COUNT; i++)
		{
			j = j ^ (j << 13);
			j = j ^ (j >> 17);
			j = j ^ (j << 5);
			m_Q[i] = j;
		}

		m_c = 362436;
		m_i = SEED_COUNT - 1;
	}


	uint32	generator::next_random()
	// Return the next pseudo-random number in the sequence.
	{
		uint64	t;
		uint32	x;

		//static uint32	c = 362436;
		//static uint32	i = SEED_COUNT - 1;
		const uint32	r = 0xFFFFFFFE;

		m_i = (m_i + 1) & (SEED_COUNT - 1);
		t = a * m_Q[m_i] + m_c;
		m_c = (uint32) (t >> 32);
		x = (uint32) (t + m_c);
		if (x < m_c)
		{
			x++;
			m_c++;
		}
		
		uint32	val = r - x;
		m_Q[m_i] = val;
		return val;
	}

	
	float	generator::get_unit_float()
	{
		uint32	r = next_random();

		// 24 bits of precision.
		return float(r >> 8) / (16777216.0f - 1.0f);
	}

}	// end namespace mwcrand
} // namespace lts


// Local Variables:
// mode: C++
// c-basic-offset: 8 
// tab-width: 8
// indent-tabs-mode: t
// End:
