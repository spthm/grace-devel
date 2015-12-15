#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Makes a bitmap from numeric-type data stored at img_src.
// It is assumed that img_src may be addressed to at least i = width * height.
// Each value from img_src is converted to an intensity in R, G and B according
// to {r,g,b}_max. That is, the i-th pixel's {r,g,b} components are,
//   (img_src[i] - img_min) * {r,g,b}_max / (img_max - img_min)
// img_max - img_min must therefore have a non-zero, finite value.
template <typename T>
void make_bitmap(const T* const img_src,
                 const size_t width,
                 const size_t height,
                 const T img_min,
                 const T img_max,
                 const std::string img_fname,
                 const unsigned char r_max = 150,
                 const unsigned char g_max = 210,
                 const unsigned char b_max = 255)
{
    const size_t filesize = 54 + 3 * width * height;
    unsigned char *img = NULL;
    img = (unsigned char*)malloc(filesize);
    memset(img, 0, filesize);

    unsigned char r, g, b;
    const T img_range = img_max - img_min;
    for(size_t i = 0; i < width; ++i)
    {
        for(size_t j = 0; j < height; ++j)
        {
            r = (img_src[i + j * width] - img_min) * r_max / img_range;
            g = (img_src[i + j * width] - img_min) * g_max / img_range;
            b = (img_src[i + j * width] - img_min) * b_max / img_range;

            r = std::max((unsigned char)0, std::min(r, r_max));
            g = std::max((unsigned char)0, std::min(g, g_max));
            b = std::max((unsigned char)0, std::min(b, b_max));

            // bitmap images are encoded BGR.
            img[(i + j * width) * 3 + 0] = b;
            img[(i + j * width) * 3 + 1] = g;
            img[(i + j * width) * 3 + 2] = r;
        }
    }

    unsigned char BITMAPFILEHEADER[14];
    memset(BITMAPFILEHEADER, 0, 14);
    // Magic number to identify BMP files.
    BITMAPFILEHEADER[0] = 0x42; // ASCII 'B'
    BITMAPFILEHEADER[1] = 0x4D; // ASCII 'M'
    // File size, little-endian.
    BITMAPFILEHEADER[2] = (filesize)       & 0xff;
    BITMAPFILEHEADER[3] = (filesize >> 8)  & 0xff;
    BITMAPFILEHEADER[4] = (filesize >> 16) & 0xff;
    BITMAPFILEHEADER[5] = (filesize >> 24) & 0xff;
    // Offset, in bytes, from start of header to image data, little-endian.
    BITMAPFILEHEADER[10] = 54;

    unsigned char BITMAPINFOHEADER[40];
    memset(BITMAPINFOHEADER, 0, 40);
    // Size of BITMAPINFOHEADER, in bytes, little-endian.
    BITMAPINFOHEADER[0] = 40;
    // Width of image, in pixels, little-endian.
    BITMAPINFOHEADER[4] =  (width)       & 0xff;
    BITMAPINFOHEADER[5] =  (width >> 8)  & 0xff;
    BITMAPINFOHEADER[6] =  (width >> 16) & 0xff;
    BITMAPINFOHEADER[7] =  (width >> 24) & 0xff;
    // Height of image, in pixels, little-endian.
    BITMAPINFOHEADER[8] =  (height)       & 0xff;
    BITMAPINFOHEADER[9] =  (height >> 8)  & 0xff;
    BITMAPINFOHEADER[10] = (height >> 16) & 0xff;
    BITMAPINFOHEADER[11] = (height >> 24) & 0xff;
    // Number of colour planes (must be 1).
    BITMAPINFOHEADER[12] = 1;
    // Bits per pixel (colour depth).
    BITMAPINFOHEADER[14] = 24;
    // Compression method Bl_RGB (no compression).
    BITMAPINFOHEADER[16] = 0;

    unsigned char pad[3] = {0, 0, 0};

    FILE* f = fopen(img_fname.c_str(), "wb");
    fwrite(BITMAPFILEHEADER, 1, 14, f);
    fwrite(BITMAPINFOHEADER, 1, 40, f);
    // The length of each row in bytes must be a multiple of 4.
    int row_pad_size = (4 - (3 * width) % 4) % 4;
    for(int i = 0; i < height; ++i)
    {
        // Bitmap images are stored left-to-right, bottom-to-top.
        unsigned char* row_start = img + 3 * width * (height - i - 1);
        fwrite(row_start, 3, width, f);
        fwrite(pad, 1, row_pad_size, f);
    }
    fclose(f);
}
