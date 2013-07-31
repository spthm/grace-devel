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

/*! \file sobol.h
 *   \brief Define Sobol samplers and sequences.
 */

#pragma once

#include <nih/basic/types.h>
#include <nih/sampling/sobol_matrices.h>
#include <limits.h>

namespace nih {

/*! \addtogroup sampling Sampling
 */

/*! \addtogroup sobol Sobol Sequences
 *  \ingroup sampling
 *  \{
 */

///
/// Sobol sampler class
///
class Sobol_sampler
{
  public:
    static const unsigned int max = UINT_MAX;

    /// empty constructor
    ///
    FORCE_INLINE NIH_HOST_DEVICE Sobol_sampler();

    /// instance constructor
    ///
    /// \param instance     instance number
    /// \param s            randomization seed
    FORCE_INLINE NIH_HOST_DEVICE Sobol_sampler(unsigned int instance, unsigned int s = 0);

    /// return next sample
    ///
    FORCE_INLINE NIH_HOST_DEVICE float sample();
    /// return next sample
    ///
    FORCE_INLINE NIH_HOST_DEVICE float next();

    /// build generator matrices
    ///
    /// \param direction_numbers    input file of comma separated direction numbers
    /// \param matrix_file          output file
    static void generator_matrices(
        const char* direction_numbers,
        const char* matrix_file);

  private:
    /// advance to next dimension
    ///
    FORCE_INLINE NIH_HOST_DEVICE void next_dim();

    /// return i-th Sobol number
    ///
    /// \param i    requested Sobol number
    FORCE_INLINE NIH_HOST_DEVICE 
    unsigned int sobol( unsigned int i );

    /// build generator matrices
    ///
    static int generator_matrix(
        const unsigned int a,
        const unsigned int s,
        const unsigned int * const m,
        unsigned int * const matrix, const unsigned int matrixSize);

    unsigned int  m_dim;
    unsigned int  m_r;
    unsigned int  m_i;
    unsigned int* s_matrix;
};

///
/// Sobol sequence interface
///
struct Sobol_sequence
{
    typedef Sobol_sampler Sampler_type;

    /// instance a Sobol sampler
    ///
    /// \param index    instance number
    /// \param copy     randomization seed
    Sobol_sampler instance(
        const uint32 index,
        const uint32 copy = 0) const { return Sobol_sampler( index, copy ); }
};

/*! \}
 */

} // namespace nih

#include <nih/sampling/sobol_inline.h>
