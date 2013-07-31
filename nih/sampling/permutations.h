/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*! \file permutations.h
 *   \brief Defines functions and classes to build pseudo-random permutations and
 *          permutation sets.
 */

#pragma once

#include <nih/basic/types.h>
#include <nih/basic/primes.h>
#include <nih/basic/numbers.h>
#include <nih/sampling/random.h>

namespace nih {

/*! \addtogroup sampling Sampling
 */

/*! \addtogroup permutations Permutations
 *  \ingroup sampling
 *  \{
 */

/// permute a set of n elements randomly
///
/// \param random   random number generator
/// \param n        set size
/// \param seq      element sequence
template <typename Sequence>
void permute(Random& random, const uint32 n, Sequence seq)
{
	for (uint32 i = 0; i < n-1; i++)
	{
		const float r = random.next();
		const uint32 j = std::min( uint32(float(r) * (n - i)) + i, n-1u );

		std::swap( seq[i], seq[j] );
	}
}

///
/// A permutation set based on linear congruences
///
struct LCPermutation_set
{
    /// Build n-permutations of the range [0,m), with n << m.
    ///
    /// \param m    range size
    /// \param n    number of permutations
    LCPermutation_set(const uint32 m, const uint32 n) :
        m_M(m), m_A(n), m_C(n)
    {
        Random random;

        // we want to pick n prime numbers at random in the range [m/2,m]
        for (uint32 i = 0; i < n; ++i)
        {
            const float r1 = (float(i) + random.next())/float(n);
            m_A[i] = s_primes[ m/2 + quantize( r1, m/2-1 ) ];

            const float r2 = (float(i) + random.next())/float(n);
            m_C[i] = s_primes[ m/2 + quantize( r2, m/2-1 ) ];
        }

        // let's permute A and C independently
        permute( random, n, m_A );
        permute( random, n, m_C );
    }

    /// return the permuted position of a given index in a given permutation
    ///
    /// \param permutation_index        permutation index
    /// \param element_index            element index
    FORCE_INLINE uint32 operator() (
        const uint32 permutation_index,
        const uint32 element_index) const
    {
        return m_A[ permutation_index ] * element_index + m_C[ permutation_index ] & (m_M-1);
    }

    uint32                   m_M;
    std::vector<nih::uint32> m_A;
    std::vector<nih::uint32> m_C;
};

///
/// A permutation set based on actual random permutation tables
///
struct Permutation_set
{
    /// Build n-permutations of the range [0,m)
    ///
    /// \param m    range size
    /// \param n    number of permutations
    Permutation_set(const uint32 m, const uint32 n) :
        m_M(m), m_tables(m*n)
    {
        Random random;

        // loop through the set of n permutations
        for (uint32 i = 0; i < n; ++i)
        {
            // build the i-th table
            for (nih::uint32 j = 0; j < m; ++j)
                m_tables[ i*m + j ] = j;

            permute( random, m, &m_tables[i*m] );
        }
    }

    /// return the permuted position of a given index in a given permutation
    ///
    /// \param permutation_index        permutation index
    /// \param element_index            element index
    FORCE_INLINE uint32 operator() (
        const uint32 permutation_index,
        const uint32 element_index) const
    {
        return m_tables[ permutation_index * m_M + element_index ];
    }

    uint32                   m_M;
    std::vector<nih::uint32> m_tables;
};

///
/// A wrapper to build permuted sequences based on a templated permutation
/// sequence
///
template <typename Permutation_sequence, typename Sample_sequence>
struct Permuted_sequence
{
    typedef typename Sample_sequence::Sampler_type Sampler_type;

    /// constructor
    ///
    /// \param sequence                sample sequence
    /// \param permutation_sequence    permutation sequence
    Permuted_sequence(
        Sample_sequence& sequence,
        Permutation_sequence& permutation_sequence) :
        m_sequence( sequence ), m_permutation( permutation ) {}

    /// return a given sampler instance
    ///
    /// \param index        instance number
    /// \param copy         randomization seed
    Sampler_type instance(const uint32 index, const uint32 copy)
    {
        return m_sequence.instance( m_permutation(copy,index), 0 );
    }

    Permutation_sequence& m_permutation;
    Sample_sequence&      m_sequence;
};

/*! \}
 */

} // namespace nih
