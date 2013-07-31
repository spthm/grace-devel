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

/*! \file image.h
 *   \brief Defines a simple 2d image interface.
 */

#ifndef __NIH_IMAGE_H
#define __NIH_IMAGE_H

#include <nih/linalg/vector.h>
#include <vector>

namespace nih {

/*! \addtogroup images Images
 *  \{
 */

///
/// An abstract image (i.e. large matrix) class over elements of templated type T.
///
template <typename T>
class Image
{
public:
	typedef T Field_type;
	typedef typename std::vector<T*>::const_iterator Row_const_iterator;
	typedef typename std::vector<T*>::iterator       Row_iterator;

	/// empty constructor
    ///
	Image() : m_n_rows(0), m_n_cols(0) {}

	/// copy constructor
    ///
    /// \param image    image to copy
	Image(const Image& image);

	/// copy operator
    ///
    /// \param image    image to copy
	Image& operator=(const Image& image);

	/// set the image resolution
    ///
    /// \param res_x    x resolution
    /// \param res_y    y resolution
    /// \param def      default value
	void set(const uint32 res_x, const uint32 res_y, const T def = T(0.0));

	/// clear the image to a default value
    ///
    /// \param def      default value
	void clear(const T def);

	/// return resolution
    ///
    /// \param i        axis
	uint32 resolution(const uint32 i)  const { return i == 0 ? cols() : rows(); }

	/// return # of columns
    ///
	uint32 cols() const { return m_n_cols; }

	/// return # of rows
    ///
	uint32 rows() const { return m_n_rows; }

	/// non-const access to (x,y) pixel
    ///
    /// \param x    x coordinate
    /// \param y    y coordinate
	T& operator() (const uint32 x, const uint32 y);

	/// const access to (x,y) pixel
    ///
    /// \param x    x coordinate
    /// \param y    y coordinate
	const T& operator() (const uint32 x, const uint32 y) const;

	/// dereference operator for rows
    ///
    /// \param y    y coordinate
	const T* operator[] (const uint32 y) const;

	/// non-const dereference operator for rows
    ///
    /// \param y    y coordinate
	T* operator[] (const uint32 y);

	/// begin row iterator
    ///
	Row_const_iterator begin() const { return m_rows.begin(); }

	/// end row iterator
    ///
	Row_const_iterator end() const { return m_rows.end(); }

    /// begin row iterator
    ///
	Row_iterator begin() { return m_rows.begin(); }

    /// begin row iterator
    ///
	Row_iterator end() { return m_rows.end(); }

private:
	/// setup the row array
	void setup_rows();

	std::vector<T>	m_pixels;
	std::vector<T*>	m_rows;
	uint32			m_n_rows;
	uint32			m_n_cols;
};

/*! \}
 */

} // namespace nih

#include <nih/image/image_inline.h>

#endif