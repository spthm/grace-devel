#ifndef S3D_UTIL_ENDIANESS_H
#define S3D_UTIL_ENDIANESS_H

namespace s3d { 

inline void endian_swap(uint16_t & x)
{
	    x = (x>>8) | 
			(x<<8);
}

inline void endian_swap(uint32_t & x)
{
	    x = (x>>24) | 
			((x<<8) & 0x00FF0000) |
			((x>>8) & 0x0000FF00) |
			(x<<24);
}

inline void endian_swap(uint64_t & x)
{
	x = (x>>56) | 
		((x<<40) & 0x00FF000000000000) |
		((x<<24) & 0x0000FF0000000000) |
		((x<<8)  & 0x000000FF00000000) |
		((x>>8)  & 0x00000000FF000000) |
		((x>>24) & 0x0000000000FF0000) |
		((x>>40) & 0x000000000000FF00) |
		(x<<56);
}

} 


#endif
