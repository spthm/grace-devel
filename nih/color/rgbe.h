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

#include <nih/linalg/vector.h>

namespace nih
{

NIH_HOST_DEVICE FORCE_INLINE uint32 toRGBE(const float r, const float g, const float b)
{
    float v = 0;
    if (r > v) v = r;
    if (g > v) v = g;
    if (b > v) v = b;

    union
    {
        float  f;
        uint32 i;
    } fi;

    fi.f = v;
    int exponent = ((fi.i >> 23u) & 0xFF) - 126u;
    int rgbe = (exponent + 128u) & 0xFF;
    if (rgbe < 10) return 0;
    fi.i = ((((rgbe & 0xFF) - (128u + 8u)) + 127u) << 23u) & 0x7F800000;

    float f = 1.0f / fi.f;
    rgbe |= ((uint32) (r * f) << 24);
    rgbe |= ((uint32) (g * f) << 16);
    rgbe |= ((uint32) (b * f) << 8);
    return rgbe;
}

NIH_HOST_DEVICE FORCE_INLINE Vector3f fromRGBE(const uint32 rgbe)
{
    union
    {
        float  f;
        uint32 i;
    } fi;

    //fi.i = ((((rgbe & 0xFF) - (128u + 8u)) + 127u) << 23u) & 0x7F800000;
    fi.i = (((rgbe & 0xFF) - 9u) << 23u) & 0x7F800000;

    const float f = fi.f;
    return Vector3f(
        f *  (rgbe >> 24),
        f * ((rgbe >> 16) & 0xFF),
        f * ((rgbe >> 8)  & 0xFF) );
}

} // namespace nih
