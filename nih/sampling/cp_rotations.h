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

/*! \file cp_rotations.h
 *   \brief Defines utility classes to perform Cranley-Patterson rotations.
 */

#pragma once

#include <nih/basic/numbers.h>

namespace nih {

/*! \addtogroup sampling Sampling
 */

/*! \addtogroup cp_rotations Cranley-Patterson Rotations
 *  \ingroup sampling
 *  \{
 */

///
/// Wrapper class to rotate the samples coming from a generator with
/// a set of Cranley-Patterson rotations.
///
template <typename Generator>
struct CP_rotator
{
    /// constructor
    ///
    /// \param gen      generator
    /// \param rot      sequence of rotations
    /// \param size     number of rotations
    /// \param dim      initial dimension
    NIH_HOST_DEVICE CP_rotator(Generator& gen, const float* rot, uint32 size, uint32 dim = 0) :
        m_gen( gen ), m_rot( rot ), m_size( size ), m_dim( dim ) {}

    /// get the next sample
    ///
	inline NIH_HOST_DEVICE float next()
	{
        return nih::mod( m_gen.next() + m_rot[m_dim++ & (m_size-1)], 1.0f );
    }
    /// probability density function
    ///
    /// \param x    sample location
	inline NIH_HOST_DEVICE float density(const float x) const
	{
        return 1.0f;
    }

    Generator&        m_gen;
    const float*      m_rot;
    uint32            m_size;
    uint32            m_dim;
};

///
/// Wrapper class to rotate the samples coming from a sample sequence with
/// a set of Cranley-Patterson rotations.
///
template <typename Sample_sequence>
struct CP_rotated_sequence
{
    typedef CP_rotator<typename Sample_sequence::Sampler_type> Sampler_type;

    /// constructor
    ///
    /// \param sequence sample sequence
    /// \param dims     dimensionality of the sequence
    /// \param rot      rotation set
    /// \param size     size of the rotation set
    NIH_HOST_DEVICE CP_rotated_sequence(Sample_sequence& sequence, const uint32 dims, const float* rot, uint32 size) :
        m_sequence( sequence ), m_rot( rot ), m_size( size ), m_dims( dims ) {}

    /// instance
    ///
    /// \param index    sequence index
    /// \param copy     sequence instance
    Sampler_type NIH_HOST_DEVICE instance(const uint32 index, const uint32 copy) const
    {
        return Sampler_type( m_sequence.instance( index, copy ), m_rot, m_dims * m_size, copy * m_dims );
    }

    Sample_sequence&      m_sequence;
    const float*          m_rot;
    uint32                m_size;
    uint32                m_dims;
};

/*! \}
 */

} // namespace nih
