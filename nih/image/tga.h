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

/*
    Very simple TGA reading/writing.
*/


#ifndef TGA_INCLUDED
#define TGA_INCLUDED

namespace nih {

//////////////////////////////////////////////////////////////////////////
// TGA File Header
//////////////////////////////////////////////////////////////////////////
#pragma pack(push,1) // no alignment
struct TGAHeader
{
    unsigned char  identsize;           // size of ID field that follows 18 byte header (0 usually)
    unsigned char  colourmaptype;       // type of colour map 0=none, 1=has palette
    unsigned char  imagetype;           // type of image 0=none,1=indexed,2=rgb,3=grey,+8=rle packed

    unsigned short colourmapstart;      // first colour map entry in palette
    unsigned short colourmaplength;     // number of colours in palette
    unsigned char  colourmapbits;       // number of bits per palette entry 15,16,24,32

    unsigned short xstart;              // image x origin
    unsigned short ystart;              // image y origin
    unsigned short width;               // image width in pixels
    unsigned short height;              // image height in pixels
    unsigned char  bits;                // image bits per pixel 8,16,24,32
    unsigned char  descriptor;          // image descriptor bits (vh flip bits)
};
#pragma pack(pop)



// Load an uncompressed tga image, 24 or 32 bpp. The pixel memory is allocated
// by the routine and must be freed by the caller using delete[].
unsigned char*  load_tga ( const char *filename, TGAHeader *hdr );

// Write a TGA to file, 24 bpp, with the specified parameters. rgb indicates
// whether pixdata is in RGB or BGR format.
bool            write_tga ( const char* filename, int width, int height, unsigned char *pixdata, bool rgb );

} // namespace nih

#endif