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

namespace nih {

// get a set of 2d stratified samples
template <typename T>
void MJSampler::sample(
	const uint32	samples_x,
	const uint32	samples_y,
	Vector<T,2>*	samples,
	Ordering		ordering)
{
	m_sample_xy.resize( samples_x * samples_y );
	for (uint32 y = 0; y < samples_y; y++)
	{
		for (uint32 x = 0; x < samples_x; x++)
		{
			const size_t index = y * samples_x + x;
			m_sample_xy[index].x = y;
			m_sample_xy[index].y = x;
		}
	}
	for (uint32 y = 0; y < samples_y; y++)
	{
		for (uint32 x = 0; x < samples_x-1; x++)
		{
			const float r = m_random.next();
			const uint32 xx = std::min( uint32(float(r) * (samples_x - x)) + x, samples_x-1u );

			const size_t index1 = y * samples_x + x;
			const size_t index2 = y * samples_x + xx;

			std::swap( m_sample_xy[index1].y, m_sample_xy[index2].y );
		}
	}
	for (uint32 x = 0; x < samples_x; x++)
	{
		for (uint32 y = 0; y < samples_y-1; y++)
		{
			const float r = m_random.next();
			const uint32 yy = std::min( uint32(float(r) * (samples_y - y)) + y, samples_y-1u );

			const size_t index1 = y * samples_x + x;
			const size_t index2 = yy * samples_x + x;

			std::swap( m_sample_xy[index1].x, m_sample_xy[index2].x );
		}
	}

	const uint32 num_samples = samples_x * samples_y;
	const T inv = T(1.0) / T(num_samples);
	for (uint32 y = 0; y < samples_y; y++)
	{
		for (uint32 x = 0; x < samples_x; x++)
		{
			const size_t index = y * samples_x + x;
			samples[index][0] = (T(m_sample_xy[index].x + x * samples_y) + m_random.next()) * inv;
			samples[index][1] = (T(m_sample_xy[index].y + y * samples_x) + m_random.next()) * inv;
		}
	}
	if (ordering == kRandom)
	{
		for (uint32 i = 0; i < num_samples; i++)
		{
			const float r = m_random.next();
			const uint32 j = std::min( uint32(float(r) * (num_samples - i)) + i, num_samples-1u );
			std::swap( samples[i], samples[j] );
		}
	}
}
// get a set of 3d stratified samples
// the first 2 dimensions are multi-jittered, the third one
// is selected with latin hypercube sampling wrt the first 2.
template <typename T>
void MJSampler::sample(
	const uint32	samples_x,
	const uint32	samples_y,
	Vector<T,3>*	samples)
{
	m_sample_xy.resize( samples_x * samples_y );
	for (uint32 y = 0; y < samples_y; y++)
	{
		for (uint32 x = 0; x < samples_x; x++)
		{
			const size_t index = y * samples_x + x;
			m_sample_xy[index].x = y;
			m_sample_xy[index].y = x;
		}
	}
	for (uint32 y = 0; y < samples_y; y++)
	{
		for (uint32 x = 0; x < samples_x-1; x++)
		{
			const float r = m_random.next();
			const uint32 xx = std::min( uint32(float(r) * (samples_x - x)) + x, samples_x-1u );

			const size_t index1 = y * samples_x + x;
			const size_t index2 = y * samples_x + xx;

			std::swap( m_sample_xy[index1].y, m_sample_xy[index2].y );
		}
	}
	for (uint32 x = 0; x < samples_x; x++)
	{
		for (uint32 y = 0; y < samples_y-1; y++)
		{
			const float r = m_random.next();
			const uint32 yy = std::min( uint32(float(r) * (samples_y - y)) + y, samples_y-1u );

			const size_t index1 = y * samples_x + x;
			const size_t index2 = yy * samples_x + x;

			std::swap( m_sample_xy[index1].x, m_sample_xy[index2].x );
		}
	}

	const uint32 num_samples = samples_x * samples_y;
	const T inv = T(1.0) / T(num_samples);
	for (uint32 y = 0; y < samples_y; y++)
	{
		for (uint32 x = 0; x < samples_x; x++)
		{
			const size_t index = y * samples_x + x;
			samples[index][0] = (T(m_sample_xy[index].x + x * samples_y) + m_random.next()) * inv;
			samples[index][1] = (T(m_sample_xy[index].y + y * samples_x) + m_random.next()) * inv;
			samples[index][2] = (T(index) + m_random.next()) * inv;
		}
	}
	for (uint32 i = 0; i < num_samples; i++)
	{
		const float r = m_random.next();
		const uint32 j = std::min( uint32(float(r) * (num_samples - i)) + i, num_samples-1u );
		std::swap( samples[i], samples[j] );
	}
}
// get a set of 4d stratified samples
template <typename T>
void MJSampler::sample(
	const uint32	samples_x,
	const uint32	samples_y,
	Vector<T,4>*	samples,
	Ordering		ordering)
{
	for (uint32 offset = 0; offset <= 2; offset += 2)
	{
		m_sample_xy.resize( samples_x * samples_y );
		for (uint32 y = 0; y < samples_y; y++)
		{
			for (uint32 x = 0; x < samples_x; x++)
			{
				const size_t index = y * samples_x + x;
				m_sample_xy[index].x = y;
				m_sample_xy[index].y = x;
			}
		}
		for (uint32 y = 0; y < samples_y; y++)
		{
			for (uint32 x = 0; x < samples_x-1; x++)
			{
				const float r = m_random.next();
				const uint32 xx = std::min( uint32(float(r) * (samples_x - x)) + x, samples_x-1u );

				const size_t index1 = y * samples_x + x;
				const size_t index2 = y * samples_x + xx;

				std::swap( m_sample_xy[index1].y, m_sample_xy[index2].y );
			}
		}
		for (uint32 x = 0; x < samples_x; x++)
		{
			for (uint32 y = 0; y < samples_y-1; y++)
			{
				const float r = m_random.next();
				const uint32 yy = std::min( uint32(float(r) * (samples_y - y)) + y, samples_y-1u );

				const size_t index1 = y * samples_x + x;
				const size_t index2 = yy * samples_x + x;

				std::swap( m_sample_xy[index1].x, m_sample_xy[index2].x );
			}
		}

		const uint32 num_samples = samples_x * samples_y;
		const T inv = T(1.0) / T(num_samples);
		for (uint32 y = 0; y < samples_y; y++)
		{
			for (uint32 x = 0; x < samples_x; x++)
			{
				const size_t index = y * samples_x + x;
				samples[index][offset+0] = (T(m_sample_xy[index].x + x * samples_y) + m_random.next()) * inv;
				samples[index][offset+1] = (T(m_sample_xy[index].y + y * samples_x) + m_random.next()) * inv;
			}
		}
		if (ordering == kRandom)
		{
			const uint32 num_samples = samples_x * samples_y;
			for (uint32 i = 0; i < num_samples; i++)
			{
				const float r = m_random.next();
				const uint32 j = std::min( uint32(float(r) * (num_samples - i)) + i, num_samples-1u );
				std::swap( samples[i][offset+0], samples[j][offset+0] );
				std::swap( samples[i][offset+1], samples[j][offset+1] );
			}
		}
	}
}

inline void Sampler::sample(
	const uint32	num_samples,
	uint32*			samples)
{
	for (uint32 i = 0; i < num_samples; i++)
		samples[i] = i;

	for (uint32 i = 0; i < num_samples; i++)
	{
		const float r = m_random.next();
		const uint32 j = std::min( uint32(float(r) * (num_samples - i)) + i, num_samples-1u );
		std::swap( samples[i], samples[j] );
	}
}

// get a set of 1d stratified samples
template <typename T>
inline void Sampler::sample(
	const uint32	num_samples,
	T*				samples)
{
	m_sample_x.resize( num_samples );

	for (uint32 i = 0; i < num_samples; i++)
		m_sample_x[i] = i;

	for (uint32 i = 0; i < num_samples; i++)
	{
		const float r = m_random.next();
		const uint32 j = std::min( uint32(float(r) * (num_samples - i)) + i, num_samples-1u );
		std::swap( m_sample_x[i], m_sample_x[j] );
	}
	const T inv = T(1.0) / T(num_samples);
	for (uint32 i = 0; i < num_samples; i++)
		samples[i] = (T(m_sample_x[i]) + m_random.next()) * inv;
}
// get a set of 2d stratified samples
template <typename T>
inline void Sampler::sample(
	const uint32	num_samples,
	Vector<T,2>*	samples)
{
	m_sample_x.resize( num_samples );
	m_sample_y.resize( num_samples );

	for (uint32 i = 0; i < num_samples; i++)
	{
		m_sample_x[i] = i;
		m_sample_y[i] = i;
	}
	for (uint32 i = 0; i < num_samples; i++)
	{
		{
			const float r = m_random.next();
			const uint32 j = std::min( uint32(float(r) * (num_samples - i)) + i, num_samples-1u );
			std::swap( m_sample_x[i], m_sample_x[j] );
		}
		{
			const float r = m_random.next();
			const uint32 j = std::min( uint32(float(r) * (num_samples - i)) + i, num_samples-1u );
			std::swap( m_sample_y[i], m_sample_y[j] );
		}
	}
	const T inv = T(1.0) / T(num_samples);
	for (uint32 i = 0; i < num_samples; i++)
	{
		samples[i][0] = (T(m_sample_x[i]) + m_random.next()) * inv;
		samples[i][1] = (T(m_sample_y[i]) + m_random.next()) * inv;
	}
}
// get a set of 3d stratified samples
template <typename T>
inline void Sampler::sample(
	const uint32	num_samples,
	Vector<T,3>*	samples)
{
	m_sample_x.resize( num_samples );

	const T inv = T(1.0) / T(num_samples);
	for (int j = 0; j < 3; j++)
	{
		for (uint32 i = 0; i < num_samples; i++)
			m_sample_x[i] = i;

		for (uint32 i = 0; i < num_samples; i++)
		{
			const float r = m_random.next();
			const uint32 k = std::min( uint32(float(r) * (num_samples - i)) + i, num_samples-1u );
			std::swap( m_sample_x[i], m_sample_x[k] );
		}
		for (uint32 i = 0; i < num_samples; i++)
			samples[i][j] = (T(m_sample_x[i]) + m_random.next()) * inv;
	}
}
// get a set of 4d stratified samples
template <typename T>
inline void Sampler::sample(
	const uint32	num_samples,
	Vector<T,4>*	samples)
{
	m_sample_x.resize( num_samples );

	const T inv = T(1.0) / T(num_samples);
	for (int j = 0; j < 4; j++)
	{
		for (uint32 i = 0; i < num_samples; i++)
			m_sample_x[i] = i;

		for (uint32 i = 0; i < num_samples; i++)
		{
			const float r = m_random.next();
			const uint32 k = std::min( uint32(float(r) * (num_samples - i)) + i, num_samples-1u );
			std::swap( m_sample_x[i], m_sample_x[k] );
		}
		for (uint32 i = 0; i < num_samples; i++)
			samples[i][j] = (T(m_sample_x[i]) + m_random.next()) * inv;
	}
}

/// get a set of N-d stratified samples
template <typename Image_type>
void Sampler::sample(
	const uint32	num_samples,
	const uint32	num_dims,
	Image_type&		samples)
{
	m_sample_x.resize( num_samples );

	typedef typename Image_type::Field_type T;

	const T inv = T(1.0) / T(num_samples);
	for (uint32 j = 0; j < num_dims; j++)
	{
		for (uint32 i = 0; i < num_samples; i++)
			m_sample_x[i] = i;

		for (uint32 i = 0; i < num_samples; i++)
		{
			const float r = m_random.next();
			const uint32 k = std::min( uint32(float(r) * (num_samples - i)) + i, num_samples-1u );
			std::swap( m_sample_x[i], m_sample_x[k] );
		}
		for (uint32 i = 0; i < num_samples; i++)
			samples[i][j] = (T(m_sample_x[i]) + m_random.next()) * inv;
	}
}

// sample a cdf
inline uint32 sample_cdf(
	const float					x,
	const std::vector<float>&	cdf,
	float&						pdf)
{
//	const uint32 index = std::min( uint32( std::lower_bound( &cdf[0], &cdf[0] + cdf.size(), x ) - &cdf[0] ), uint32(cdf.size()-1u) );
//	pdf = index == 0 ? cdf[0] : cdf[index] - cdf[index-1];
//	return index;
	uint32 index = std::min( uint32( std::lower_bound( &cdf[0], &cdf[0] + cdf.size(), x ) - &cdf[0] ), uint32(cdf.size()-1u) );
    do
    {
    	pdf = index == 0 ? cdf[0] : cdf[index] - cdf[index-1];
        if (pdf > 0.0f)
            break;
    }
    while(++index < cdf.size());

    return index;
}
// sample a cdf
inline uint32 sample_cdf(
	const double				x,
	const std::vector<double>&	cdf,
	double&						pdf)
{
//	const uint32 index = std::min( uint32( std::lower_bound( &cdf[0], &cdf[0] + cdf.size(), x ) - &cdf[0] ), uint32(cdf.size()-1u) );
//	pdf = index == 0 ? cdf[0] : cdf[index] - cdf[index-1];
//	return index;
	uint32 index = std::min( uint32( std::lower_bound( &cdf[0], &cdf[0] + cdf.size(), x ) - &cdf[0] ), uint32(cdf.size()-1u) );
    do
    {
    	pdf = index == 0 ? cdf[0] : cdf[index] - cdf[index-1];
        if (pdf > 0.0f)
            break;
    }
    while(++index < cdf.size());

    return index;
}
// sample a cdf
inline uint32 sample_cdf(
	const float					x,
	const uint32				n,
	const float*				cdf,
	float&						pdf)
{
	const uint32 index = std::min( uint32( std::lower_bound( &cdf[0], &cdf[0] + n, x ) - &cdf[0] ), uint32(n-1u) );
	pdf = index == 0 ? cdf[0] : cdf[index] - cdf[index-1];
	return index;
}

// sample a cdf
inline float sample_cdf_cont(
	const float					x,
	const std::vector<float>&	cdf,
	float&						pdf)
{
	const uint32 index = std::min( uint32( std::lower_bound( &cdf[0], &cdf[0] + cdf.size(), x ) - &cdf[0] ), uint32(cdf.size()-1u) );
	const float prev_cdf = index > 0 ? cdf[index-1] : 0.0f; 
	const float rem = (x - prev_cdf) / (cdf[index] - prev_cdf);
	pdf = (cdf[index] - prev_cdf) * float(cdf.size());
	return (float(index) + rem) / float(cdf.size());
}

} // namespace nih
