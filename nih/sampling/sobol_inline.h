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

#pragma once

namespace nih {

FORCE_INLINE NIH_HOST_DEVICE Sobol_sampler::Sobol_sampler()
{
#ifdef __CUDACC__
    #if (__CUDA_ARCH__ > 0)
    s_matrix = c_sobolMat;
    #else
    s_matrix = s_sobolMat;
    #endif
#else
    s_matrix = s_sobolMat;
#endif
}

FORCE_INLINE NIH_HOST_DEVICE Sobol_sampler::Sobol_sampler(unsigned int instance, unsigned int s) : m_dim( unsigned int(-1) ), m_r(s), m_i(instance)
{
#ifdef __CUDACC__
    #if (__CUDA_ARCH__ > 0)
    s_matrix = c_sobolMat;
    #else
    s_matrix = s_sobolMat;
    #endif
#else
    s_matrix = s_sobolMat;
#endif
}

FORCE_INLINE NIH_HOST_DEVICE float Sobol_sampler::sample()
{
  next_dim();
  return float( sobol( m_i ) ) / float(UINT_MAX);
}
FORCE_INLINE NIH_HOST_DEVICE float Sobol_sampler::next() { return sample(); } // random number generator interface

FORCE_INLINE NIH_HOST_DEVICE void Sobol_sampler::next_dim()
{
  ++m_dim;
  m_r = m_r * 1103515245 + 12345;
}

FORCE_INLINE NIH_HOST_DEVICE 
unsigned int Sobol_sampler::sobol( unsigned int i )
{
  unsigned int m = (m_dim & (sobolDims-1)) << 5;
  unsigned int result = 0;

  for( ; i; i >>= 1, ++m )
    result ^= s_matrix[m /*& (sobolDims*32-1)*/] * (i&1);

  result ^= m_r;
  return result;
}

inline
int Sobol_sampler::generator_matrix(
    const unsigned int a,
    const unsigned int s,
    const unsigned int * const m,
    unsigned int * const matrix, const unsigned int matrixSize)
{
    // determine degree of polynomial (could be optimized using __builtin_clz)
    //int s = 0;
    //for (int p = a >> 1; p; p >>= 1, s++);

    // first columns correspond to m_1,...,m_s
    for (unsigned int k = 0; k < s; ++k)
        matrix[k] = m[k];

    // the remaining direction numbers are obtained by recurrence
    for (unsigned int k = s; k < matrixSize; k++) {
        matrix[k] = (matrix[k - s] << s) ^ matrix[k - s];
        // iterate over bits of polynomial
        for (int i = int(s) - 1, p = a >> 1; i > 0; i--, p >>= 1)
        {
            if (p & 1)
                matrix[k] ^= (matrix[k - i] << i);
        }
    }
    // k-th column is the binary expansion of m_k / 2^k
    for (unsigned int k = 0; k < matrixSize; k++)
        matrix[k] <<= matrixSize - k - 1;

    return s;
}

} // namespace nih
