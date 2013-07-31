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

/*! \file mappings.h
 *   \brief Defines various spherical mappings.
 */

#pragma once

#include <nih/basic/numbers.h>
#include <nih/linalg/vector.h>
#include <algorithm>

namespace nih {

/*! \addtogroup spherical_functions Spherical
 */

/*! \addtogroup spherical_mappings Spherical Mappings
 *  \ingroup spherical_functions
 *  \{
 */

/// maps a point in spherical coordinates to the unit sphere
///
/// \param uv   uv coordinates
NIH_HOST NIH_DEVICE Vector3f from_spherical_coords(const Vector2f& uv);

/// computes the spherical coordinates of a 3d point
///
/// \param vec  3d direction
NIH_HOST NIH_DEVICE Vector2f to_spherical_coords(const Vector3f& vec);

/// map a point on [0,1]^2 to a uniformly distributed point on a disk of radius 1
///
/// \param uv   uv coordinates
NIH_HOST NIH_DEVICE Vector2f square_to_unit_disk(const Vector2f uv);

/// diskx, disky is point on radius 1 disk.  x, y is point on [0,1]^2
///
/// \param disk     disk point
NIH_HOST NIH_DEVICE Vector2f unit_disk_to_square(const Vector2f disk);

/// maps the unit square to the hemisphere with a cosine-weighted distribution
///
/// \param uv   uv coordinates
NIH_HOST NIH_DEVICE Vector3f square_to_cosine_hemisphere(const Vector2f& uv);

/// inverts the square to cosine-weighted hemisphere mapping
///
/// \param dir  3d direction
NIH_HOST NIH_DEVICE Vector2f cosine_hemisphere_to_square(const Vector3f& dir);

/// maps the unit square to the sphere with a uniform distribution
///
/// \param uv   uv coordinates
NIH_HOST NIH_DEVICE Vector3f uniform_square_to_sphere(const Vector2f& uv);

/// maps the sphere to a unit square with a uniform distribution
///
/// \param dir  3d direction
NIH_HOST NIH_DEVICE Vector2f uniform_sphere_to_square(const Vector3f& vec);

/*! \}
 */

} // namespace nih

#include <nih/spherical/mappings_inline.h>