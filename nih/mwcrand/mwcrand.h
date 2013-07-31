// mwcrand.h	-- Thatcher Ulrich 2003

// This source code has been donated to the Public Domain.  Do
// whatever you want with it.

// Pseudorandom number generator.


#ifndef _MWCRAND_H
#define _MWCRAND_H


#include <base/types.h>

namespace lts {

namespace mwcrand
{
	const int	SEED_COUNT = 32;
	
	// In case you need independent generators.  The global
	// generator is just an instance of this.
	struct generator
	{
		generator();
		void	seed_random(uint32 seed);	// not necessary
		uint32	next_random();
		float	get_unit_float();

	private:
		uint32	m_Q[SEED_COUNT];
		uint32	m_c;
		uint32	m_i;
	};

}	// end namespace tu_random

} // namespace lts

#endif // _MWCRAND_H


// Local Variables:
// mode: C++
// c-basic-offset: 8 
// tab-width: 8
// indent-tabs-mode: t
// End:
