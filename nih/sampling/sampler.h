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

/*! \file sampler.h
 *   \brief Defines several multidimensional samplers.
 *
 * This module provides several multi-dimensional samplers, ranging from
 * latin hypercube to multi-jittered.
 * It also provides helper objects to combine sample sets either by
 * layering of different sets for different dimensions, or by CP-rotations.
 * A sample set is represented as an image I, such that I[i][d] represents
 * the d-th coordinate of the i-th sample.
 */

#ifndef __NIH_SAMPLER_H
#define __NIH_SAMPLER_H

#include <nih/linalg/vector.h>
#include <nih/sampling/random.h>
#include <vector>
#include <algorithm>


namespace nih {

/*! \addtogroup sampling Sampling
 *  \{
 */

///
/// Return samples from a strided-vector
///
template <typename Iterator>
struct Sample_vector
{
    Sample_vector(Iterator samples, const uint32 stride) :
        m_samples( samples ), m_stride( stride ) {}

    float next()
    {
        const float value = *m_samples;
        m_samples += m_stride;
        return value;
    }

    Iterator m_samples;
    uint32   m_stride;
};

/// sample a cdf
///
/// \param x    input value
/// \param cdf  cdf to sample from
/// \param pdf  output pdf
uint32 sample_cdf(
	const float					x,
	const std::vector<float>&	cdf,
	float&						pdf);

/// sample a cdf
///
/// \param x    input value
/// \param cdf  cdf to sample from
/// \param pdf  output pdf
uint32 sample_cdf(
	const float					x,
	const uint32				n,
	const float*				cdf,
	float&						pdf);

/// sample a cdf with continuous uniform distribution inside the bins
///
/// \param x    input value
/// \param cdf  cdf to sample from
/// \param pdf  output pdf
float sample_cdf_cont(
	const float					x,
	const std::vector<float>&	cdf,
	float&						pdf);

///
/// Latin Hypercube Sampler
///
struct Sampler
{
	/// get a set of 1d stratified samples
    ///
	void sample(
		const uint32	num_samples,
		uint32*			samples);

	/// get a set of 1d stratified samples
    ///
	template <typename T>
	void sample(
		const uint32	num_samples,
		T*				samples);

	/// get a set of 2d stratified samples
    ///
	template <typename T>
	void sample(
		const uint32	num_samples,
		Vector<T,2>*	samples);

	/// get a set of 3d stratified samples
    ///
	template <typename T>
	void sample(
		const uint32	num_samples,
		Vector<T,3>*	samples);

	/// get a set of 4d stratified samples
    ///
	template <typename T>
	void sample(
		const uint32	num_samples,
		Vector<T,4>*	samples);

	/// get a set of N-d stratified samples
    ///
	template <typename Image_type>
	void sample(
		const uint32	num_samples,
		const uint32	num_dims,
		Image_type&		samples);

	std::vector<uint32>	m_sample_x;
	std::vector<uint32>	m_sample_y;
	Random				m_random;
};

///
/// Multi-Jittered Sampler
///
struct MJSampler
{
	enum Ordering
	{
		kXY,
		kRandom
	};

	/// get a set of 2d stratified samples
    ///
	template <typename T>
	void sample(
		const uint32	samples_x,
		const uint32	samples_y,
		Vector<T,2>*	samples,
		Ordering		ordering = kRandom);

	/// get a set of 3d stratified samples
	/// the first 2 dimensions are multi-jittered, the third one
	/// is selected with latin hypercube sampliing wrt the first 2.
	template <typename T>
	void sample(
		const uint32	samples_x,
		const uint32	samples_y,
		Vector<T,3>*	samples);

	/// get a set of 4d stratified samples
    ///
	template <typename T>
	void sample(
		const uint32	samples_x,
		const uint32	samples_y,
		Vector<T,4>*	samples,
		Ordering		ordering = kRandom);

	struct Sample
	{
		uint32 x;
		uint32 y;
	};
	std::vector<Sample> m_sample_xy;
	Random				m_random;
};


///
/// Combine two sets of samples (represented as images whose rows are the d-dimensional samples)
/// using CP-rotations
///
template <typename Image_type>
class Sample_combiner
{
public:
	struct Row
	{
		/// constructor
		Row(const Sample_combiner* sc, const uint32 x, const uint32 y) :
			m_sc( sc ), m_x( x ), m_y( y ), m_off( 0u ) {}

		/// constructor
		Row(const Sample_combiner* sc, const uint32 x, const uint32 y, const uint32 off) :
			m_sc( sc ), m_x( x ), m_y( y ), m_off( off ) {}

		/// return the d-th component of the row
		float operator[] (const uint32 d) const { return (*m_sc)( m_x, m_y, m_off + d ); }

        // this operator is very fishy: it makes the Row act as a float*, rather than as an iterator
		Row operator+ (const uint32 off) const {
			return Row( m_sc, m_x, m_y, m_off + off );
		}

		uint32 m_x;
		uint32 m_y;
		uint32 m_off;
		const Sample_combiner* m_sc;
	};

	/// constructor
    ///
	Sample_combiner() : m_X(NULL), m_Y(NULL) {}

	/// constructor
    ///
	Sample_combiner(
		const Image_type& X,
		const Image_type& Y)
		: m_X( &X ), m_Y( &Y )
	{
		m_X_res = m_X->rows();
		m_Y_res = m_Y->rows();
		m_dim   = std::max( m_X->cols(), m_Y->cols() );
	}

	/// return rows
    ///
	uint32 rows() const { return size(); }

	/// return cols
    ///
	uint32 cols() const { return m_dim; }

	/// size of the primary set
    ///
	uint32 primary_size() const { return m_X_res; }

	/// size of the secondary set
    ///
	uint32 secondary_size() const { return m_Y_res; }

	/// size of the combined sample set
    ///
	uint32 size() const { return m_X_res * m_Y_res; }

	/// return d-th component of the i-th sample
    ///
	float operator() (const uint32 d, const uint32 i) const
	{
		const uint32 x = i % m_X_res;
		const uint32 y = i / m_X_res;

		return operator()(x,y,d);
	}
	/// return d-th component of the (x,y)-th sample
    ///
	float operator() (const uint32 x, const uint32 y, const uint32 d) const
	{
		const float z = (d < m_Y->cols()) ? (*m_Y)[y][d] : 0.0f;
		return fmodf( (*m_X)[x][d] + z, 1.0f );
	}
	/// return d-th component of the i-th sample
    ///
	Row operator[] (const uint32 i) const
	{
		const uint32 x = i % m_X_res;
		const uint32 y = i / m_X_res;
		return Row( this, x, y );
	}

private:
	uint32				m_X_res;
	uint32				m_Y_res;
	uint32				m_dim;
	const Image_type*	m_X;
	const Image_type*	m_Y;
};

///
/// Combine two sets of samples for different dimensions
///
template <typename Image_type1, typename Image_type2>
class Sample_layer
{
public:
	struct Row
	{
		/// constructor
		Row(const Sample_layer* sc, const uint32 x) :
			m_sc( sc ), m_x( x ), m_off( 0u ) {}

    	/// constructor
		Row(const Sample_layer* sc, const uint32 x, const uint32 off) :
			m_sc( sc ), m_x( x ), m_off( off ) {}

		/// return the d-th component of the row
		float operator[] (const uint32 d) const { return (*m_sc)( m_off + d, m_x ); }

        // this operator is very fishy: it makes the Row act as a float*, rather than as an iterator
		Row operator+ (const uint32 off) const {
			return Row( m_sc, m_x, m_off + off );
		}

		uint32 m_x;
		uint32 m_off;
		const Sample_layer* m_sc;
	};

	/// constructor
	Sample_layer() : m_X(NULL), m_Y(NULL) {}

	/// constructor
	Sample_layer(
		const Image_type1& X,
		const Image_type2& Y)
		: m_X( &X ), m_Y( &Y )
	{
		m_X_dim = m_X->cols();
		m_Y_dim = m_Y->cols();
		m_res   = m_X->rows();
	}

	/// return rows
    ///
	uint32 rows() const { return size(); }

	/// return cols
    ///
	uint32 cols() const { return m_X_dim + m_Y_dim; }

	/// size of the combined sample set
    ///
	uint32 size() const { return m_res; }

	/// return d-th component of the i-th sample
    ///
	float operator() (const uint32 d, const uint32 i) const
	{
		return d < m_X_dim ? (*m_X)(d,i) : (*m_Y)(d,i);
	}
	/// return the i-th sample
    ///
	Row operator[] (const uint32 i) const
	{
		return Row( this, i );
	}

private:
	uint32				m_res;
	uint32				m_X_dim;
	uint32				m_Y_dim;
	const Image_type1*	m_X;
	const Image_type2*	m_Y;
};

///
/// Show a submatrix of a sample image
///
template <typename Image_type>
class Sample_window
{
public:
	struct Row
	{
		/// constructor
		Row(const Sample_window* sc, const uint32 y, const uint32 x = 0) :
			m_sc( sc ), m_y( y ), m_x( x ) {}

		/// return the d-th component of the row
		float operator[] (const uint32 d) const { return (*m_sc)( m_x + d, m_y ); }

        // this operator is very fishy: it makes the Row act as a float*, rather than as an iterator
		Row operator+ (const uint32 off) const {
			return Row( m_sc, m_y, m_x + off );
		}

        uint32                  m_y;
        uint32                  m_x;
		const Sample_window*    m_sc;
	};

	/// constructor
    ///
	Sample_window() : m_X(NULL), m_Y(NULL) {}

	/// constructor
    ///
	Sample_window(
		const Image_type& X,
        const uint32 x_min,
        const uint32 x_max,
        const uint32 y_min,
        const uint32 y_max) :
        m_X( &X ),
        m_x_min( x_min ),
        m_x_max( x_max ),
        m_y_min( y_min ),
        m_y_max( y_max )
	{
        m_cols = m_x_max - m_x_min + 1u;
        m_rows = m_y_max - m_y_min + 1u;
	}

	/// return rows
    ///
	uint32 rows() const { return m_rows; }

	/// return cols
    ///
	uint32 cols() const { return m_cols; }

	/// return rows
    ///
	uint32 size() const { return m_rows; }

	/// return d-th component of the i-th sample
    ///
	float operator() (const uint32 d, const uint32 i) const
	{
		return (*m_X)(d + m_x_min, i + m_y_min);
	}
	/// return the i-th sample
    ///
	Row operator[] (const uint32 i) const
	{
		return Row( this, i );
	}

private:
    uint32              m_x_min;
    uint32              m_x_max;
    uint32              m_y_min;
    uint32              m_y_max;
	uint32				m_cols;
	uint32				m_rows;
	const Image_type*	m_X;
};

///
/// Abstract sample transformation interface
///
struct Sample_transformation
{
    /// virtual destructor
    ///
    virtual ~Sample_transformation() {}

    /// warp a point
    ///
    virtual float transform(float* pt) const { return 1.0f; }

    /// compute the pdf of a given point
    ///
    virtual float density(float* pt) const { return 1.0f; }
};

/*! \}
 */

} // namespace nih

#include <nih/sampling/sampler_inline.h>

#endif