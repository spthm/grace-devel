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

#include <nih/image/tga.h>
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

namespace nih {

/************************************************************************/
/*                                                                      */
/************************************************************************/
unsigned char* load_tga ( const char *filename, TGAHeader *hdr )
{
    FILE *fp;
    if ( !(fp=fopen(filename,"rb")) )
    {
        return 0;
    }

    // read header
    fread ( hdr, sizeof(TGAHeader), 1, fp );

    // skip ident
    for(int i=0;i<hdr->identsize;i++) 
        fgetc(fp);

    // make sure we have a nice tga..
    if ( hdr->imagetype != 2 || hdr->bits != 24 && hdr->bits != 32 )
    {
        fclose ( fp );
        return 0;
    }

    // Read image data
    int bytespp = hdr->bits>>3;
    unsigned char *pixdata = new unsigned char[hdr->width*hdr->height*bytespp];
    fread ( pixdata, hdr->width*hdr->height*bytespp, 1, fp );
    fclose ( fp );

    // BGR to RGB
    for ( int i=0;i<hdr->width*hdr->height;i++ )
    {
        char c = pixdata[i*bytespp+0];
        pixdata[i*bytespp+0] = pixdata[i*bytespp+2];
        pixdata[i*bytespp+2] = c;
    }

    return pixdata;
}


/************************************************************************/
/*                                                                      */
/************************************************************************/
bool write_tga ( const char* filename, int width, int height, unsigned char *pixdata, bool rgb )
{
    FILE *fp;
    if ( !(fp=fopen(filename,"wb")) )
        return false;

    unsigned short w = (unsigned short)width;
    unsigned short h = (unsigned short)height;

    fputc ( 0, fp );    // identsize
    fputc ( 0, fp );    // colormaptype
    fputc ( 2, fp );    // imagetype
    fputc ( 0, fp );    // colormapstart
    fputc ( 0, fp );    // colormapstart
    fputc ( 0, fp );    // colormaplength
    fputc ( 0, fp );    // colormaplength
    fputc ( 0, fp );    // colormapbits
    fputc ( 0, fp );    // xstart
    fputc ( 0, fp );    // xstart
    fputc ( 0, fp );    // ystart
    fputc ( 0, fp );    // ystart
    fwrite ( &w, sizeof(unsigned short), 1, fp );   // width
    fwrite ( &h, sizeof(unsigned short), 1, fp );   // height
    fputc ( 24, fp );   // bits
    fputc ( 0, fp );    // descriptor


    if ( rgb )
    {
        // Write RGB -> BGR
        for ( int i=0; i<width*height; i++ )
        {
            fwrite ( &pixdata[i*3+2], 1, 1, fp );
            fwrite ( &pixdata[i*3+1], 1, 1, fp );
            fwrite ( &pixdata[i*3+0], 1, 1, fp );
        }
    }
    else
    {
        fwrite ( pixdata, 1, width*height*3, fp );
    }

    fclose ( fp );
    return true;
}

} // namespace nih
