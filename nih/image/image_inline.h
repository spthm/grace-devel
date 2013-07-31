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

// copy constructor
template <typename T>
Image<T>::Image(const Image& image)
{
	*this = image;
}

// copy operator
template <typename T>
Image<T>& Image<T>::operator=(const Image& image)
{
	m_n_rows = image.m_n_rows;
	m_n_cols = image.m_n_cols;

	m_pixels = image.m_pixels;

	setup_rows();
	return *this;
}

// set the image resolution
template <typename T>
void Image<T>::set(const uint32 res_x, const uint32 res_y, const T def)
{
	if (res_x != m_n_cols || res_y != m_n_rows)
	{
		m_n_rows = res_y;
		m_n_cols = res_x;

		m_pixels.resize( res_x * res_y, def );

		setup_rows();
	}
}

// clear the image to a default value
template <typename T>
void Image<T>::clear(const T def)
{
	for (uint32 i = 0; i < m_pixels.size(); i++)
		m_pixels[i] = def;
}


// non-const access to (x,y) pixel
template <typename T>
T& Image<T>::operator() (const uint32 x, const uint32 y)
{
	return m_rows[y][x];
}
// const access to (x,y) pixel
template <typename T>
const T& Image<T>::operator() (const uint32 x, const uint32 y) const
{
	return m_rows[y][x];
}

// dereference operator for rows
template <typename T>
const T* Image<T>::operator[] (const uint32 y) const
{
	return m_rows[y];
}
// non-const dereference operator for rows
template <typename T>
T* Image<T>::operator[] (const uint32 y)
{
	return m_rows[y];
}

// setup the row array
template <typename T>
void Image<T>::setup_rows()
{
	m_rows.resize( m_n_rows );
	if (m_n_cols && m_n_rows)
	{
		for (uint32 i = 0; i < m_n_rows; i++)
			m_rows[i] = &m_pixels[ i * m_n_cols ];
	}
	else
	{
		for (uint32 i = 0; i < m_n_rows; i++)
			m_rows[i] = NULL;
	}
}

} // namespace nih
