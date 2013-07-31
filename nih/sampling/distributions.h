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

/*! \file distributions.h
 *   \brief Defines various distributions.
 */

#ifndef __NIH_DISTRIBUTIONS_H
#define __NIH_DISTRIBUTIONS_H

#include <nih/sampling/random.h>

namespace nih {

/*! \addtogroup sampling Sampling
 */

/*! \addtogroup distributions Distributions
 *  \ingroup sampling
 *  \{
 */

///
/// Base distribution class
///
template <typename Derived_type>
struct Base_distribution
{
    /// return the next number in the sequence mapped through the distribution
    ///
    /// \param gen      random number generator
    template <typename Generator>
	inline float next(Generator& gen) const
	{
        return static_cast<const Derived_type*>(this)->map( gen.next() );
    }
};

///
/// Uniform distribution in [0,1]
///
struct Uniform_distribution : Base_distribution<Uniform_distribution>
{
    /// constructor
    ///
    /// \param b    distribution parameter
	NIH_HOST_DEVICE Uniform_distribution(const float b) : m_b( b ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline NIH_HOST_DEVICE float map(const float U) const { return m_b * (U*2.0f - 1.0f); }

    /// probability density function
    ///
    /// \param x    sample location
    inline NIH_HOST_DEVICE float density(const float x) const
    {
        return (x >= -m_b && x <= m_b) ? m_b * 2.0f : 0.0f;
    }

private:
    float m_b;
};
///
/// Cosine distribution
///
struct Cosine_distribution : Base_distribution<Cosine_distribution>
{
    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline NIH_HOST_DEVICE float map(const float U) const
    {
        return asin( 0.5f * U ) * 2.0f / M_PIf;
    }
    /// probability density function
    ///
    /// \param x    sample location
    inline NIH_HOST_DEVICE float density(const float x) const
    {
        if (x >= -1.0f && x <= 1.0f)
            return M_PIf*0.25f * cosf( x * M_PIf*0.5f );
        return 0.0f;
    }
};
///
/// Pareto distribution
///
struct Pareto_distribution : Base_distribution<Pareto_distribution>
{
    /// constructor
    ///
    /// \param a    distribution parameter
    /// \param min  distribution parameter
	NIH_HOST_DEVICE Pareto_distribution(const float a, const float min) : m_a( a ), m_inv_a( 1.0f / a ), m_min( min ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline NIH_HOST_DEVICE float map(const float U) const
    {
        return U < 0.5f ?
             m_min / powf( (0.5f - U)*2.0f, m_inv_a ) :
            -m_min / powf( (U - 0.5f)*2.0f, m_inv_a );
    }
    /// probability density function
    ///
    /// \param x    sample location
    inline NIH_HOST_DEVICE float density(const float x) const
    {
        if (x >= -m_min && x <= m_min)
            return 0.0f;

        return 0.5f * m_a * powf( m_min, m_a ) / powf( fabsf(x), m_a + 1.0f );
    }

private:
    float m_a;
	float m_inv_a;
    float m_min;
};
///
/// Bounded Pareto distribution
///
struct Bounded_pareto_distribution : Base_distribution<Bounded_pareto_distribution>
{
    /// constructor
    ///
    /// \param a    distribution parameter
    /// \param min  distribution parameter
    /// \param max  distribution parameter
	NIH_HOST_DEVICE Bounded_pareto_distribution(const float a, const float min, const float max) :
        m_a( a ), m_inv_a( 1.0f / a ), m_min( min ), m_max( max ),
        m_min_a( powf( m_min, m_a ) ),
        m_max_a( powf( m_max, m_a ) ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline NIH_HOST_DEVICE float map(const float U) const
    {
        if (U < 0.5f)
        {
            const float u = (0.5f - U)*2.0f;
            return powf( -(u * m_max_a - u * m_min_a - m_max_a) / (m_max_a*m_min_a), -m_inv_a);
        }
        else
        {
            const float u = (U - 0.5f)*2.0f;
            return -powf( -(u * m_max_a - u * m_min_a - m_max_a) / (m_max_a*m_min_a), -m_inv_a);
        }
    }
    /// probability density function
    ///
    /// \param x    sample location
    inline NIH_HOST_DEVICE float density(const float x) const
    {
        if (x >= -m_min && x <= m_min)
            return 0.0f;
        if (x <= -m_max || x >= m_max)
            return 0.0f;

        return 0.5f * m_a * m_min_a * powf( fabsf(x), -m_a - 1.0f ) / (1.0f - m_min_a / m_max_a);
    }

private:
    float m_a;
	float m_inv_a;
    float m_min;
    float m_max;
    float m_min_a;
    float m_max_a;
};
///
/// Bounded exponential distribution
///
struct Bounded_exponential : Base_distribution<Bounded_exponential>
{
    /// constructor
    ///
    /// \param b    distribution parameter
	NIH_HOST_DEVICE Bounded_exponential(const float b) :
		m_s1( b / 16.0f ),
		m_s2( b ),
		m_ln( -log(m_s2/m_s1) ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline NIH_HOST_DEVICE float map(const float U) const
	{
		return U < 0.5f ?
			+m_s2 * exp( m_ln*(0.5f - U)*2.0f ) :
			-m_s2 * exp( m_ln*(U - 0.5f)*2.0f );
	}

private:
    float m_s1;
	float m_s2;
	float m_ln;
};
///
/// Cauchy distribution
///
struct Cauchy_distribution : Base_distribution<Cauchy_distribution>
{
    /// constructor
    ///
    /// \param gamma    distribution parameter
	NIH_HOST_DEVICE Cauchy_distribution(const float gamma) :
		m_gamma( gamma ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline NIH_HOST_DEVICE float map(const float U) const
	{
		return m_gamma * tanf( float(M_PI) * (U - 0.5f) );
	}
    /// probability density function
    ///
    /// \param x    sample location
	inline NIH_HOST_DEVICE float density(const float x) const
	{
		return (m_gamma / (x*x + m_gamma*m_gamma)) / float(M_PI);
	}

private:
    float m_gamma;
};
///
/// Exponential distribution
///
struct Exponential_distribution : Base_distribution<Exponential_distribution>
{
    /// constructor
    ///
    /// \param lambda   distribution parameter
	NIH_HOST_DEVICE Exponential_distribution(const float lambda) :
		m_lambda( lambda ) {}

    /// transform a uniformly distributed number through the distribution
    ///
    /// \param U    real number to transform
	inline NIH_HOST_DEVICE float map(const float U) const
	{
		const float eps = 1.0e-5f;
		return U < 0.5f ?
			-logf( std::max( (0.5f - U)*2.0f, eps ) ) / m_lambda :
			 logf( std::max( (U - 0.5f)*2.0f, eps ) ) / m_lambda;
	}
    /// probability density function
    ///
    /// \param x    sample location
	inline NIH_HOST_DEVICE float density(const float x) const
	{
		return 0.5f * m_lambda * expf( -m_lambda * fabsf(x) );
	}

private:
    float m_lambda;
};
///
/// 2d Gaussian distribution
///
struct Gaussian_distribution_2d
{
    /// constructor
    ///
    /// \param sigma    variance
	NIH_HOST_DEVICE Gaussian_distribution_2d(const float sigma) :
		m_sigma( sigma ) {}

    /// transform a uniformly distributed vector through the distribution
    ///
    /// \param uv   real numbers to transform
	inline NIH_HOST_DEVICE Vector2f map(const Vector2f uv) const
	{
		const float eps = 1.0e-5f;
		const float r = m_sigma * sqrtf( - 2.0f * logf( std::max( uv[0], eps ) ) );
		return Vector2f(
			r * cosf( 2.0f * float(M_PI) * uv[1] ),
			r * sinf( 2.0f * float(M_PI) * uv[1] ) );
	}

private:
	float m_sigma;
};

///
/// Wrapper class to transform a random number generator with a given distribution
///
template <typename Generator, typename Distribution>
struct Transform_generator
{
    /// constructor
    ///
    /// \param gen      generator to wrap
    /// \param dist     transforming distribution
    NIH_HOST_DEVICE Transform_generator(Generator& gen, const Distribution& dist) : m_gen( gen ), m_dist( dist ) {}

    /// return the next number in the sequence
    ///
	inline NIH_HOST_DEVICE float next() const
	{
        return m_dist.map( m_gen.next() );
    }
    /// probability density function
    ///
    /// \param x    sample location
	inline NIH_HOST_DEVICE float density(const float x) const
	{
        return m_dist.density( x );
    }

    Generator&   m_gen;
    Distribution m_dist;
};

///
/// Wrapper class to generate Gaussian distributed numbers out of a uniform
/// random number generator
///
struct Gaussian_generator
{
    /// constructor
    ///
    /// \param sigma    variance
	NIH_HOST_DEVICE Gaussian_generator(const float sigma) :
		m_sigma( sigma ),
		m_cached( false ) {}

    /// return the next number in the sequence
    ///
    /// \param random   random number generator
    template <typename Generator>
	inline NIH_HOST_DEVICE float next(Generator& random)
	{
		if (m_cached)
		{
			m_cached = false;
			return m_cache;
		}

		const Vector2f uv( random.next(), random.next() );
		const float eps = 1.0e-5f;
		const float r = m_sigma * sqrtf( - 2.0f * logf( std::max( uv[0], eps ) ) );
		const float y0 = r * cosf( 2.0f * float(M_PI) * uv[1] );
		const float y1 = r * sinf( 2.0f * float(M_PI) * uv[1] );

		m_cache  = y1;
		m_cached = true;
		return y0;
	}
    /// probability density function
    ///
    /// \param x    sample location
    inline NIH_HOST_DEVICE float density(const float x) const
    {
        const float SQRT_TWO_PI = sqrtf(2.0f * M_PIf);
        return expf( -x*x/(2.0f*m_sigma*m_sigma) ) / (SQRT_TWO_PI*m_sigma);
    }

private:
    float  m_sigma;
	float  m_cache;
	bool   m_cached;
};

/*! \}
 */

} // namespace nih

#endif