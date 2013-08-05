#include "../pch.h"
#include "format.h"

namespace s3d { namespace img
{

const std::string &to_string(image_format fmt)/*{{{*/
{
	switch(fmt)
	{
	case BITMAP:
		{
			static const std::string ext = "bmp";
			return ext;
		}
	case KRO:
		{
			static const std::string ext = "kro";
			return ext;
		}
	case JPEG:
		{
			static const std::string ext = "jpg";
			return ext;
		}
	case PNG:
		{
			static const std::string ext = "png";
			return ext;
		}
	}

	assert(false);
	throw std::runtime_error("Invalid image format code");
}/*}}}*/

std::ostream &operator<<(std::ostream &out, const image_format &fmt)
{
	return out << to_string(fmt);
}

std::istream &operator>>(std::istream &in, image_format &fmt)
{
	std::string name;
	in >> name;
	if(!in)
		return in;

	boost::to_lower(name);

	if(name == "jpg" || name == "jpeg")
		fmt = JPEG;
	else if(name == "png")
		fmt = PNG;
	else if(name == "bmp" || name == "bitmap")
		fmt = BITMAP;
	else if(name == "kro")
		fmt = KRO;
	else
		in.clear(std::ios::failbit);

	return in;
}

}}
