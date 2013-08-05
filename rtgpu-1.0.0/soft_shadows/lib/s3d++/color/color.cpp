#include "color.h"

namespace s3d { namespace color {

std::ostream &operator<<(std::ostream &out, const model &s)
{
	switch(s)
	{
	case model::UNDEFINED:
		out << "undefined";
		break;
	case model::RADIANCE:
		out << "radiance";
		break;
	case model::GRAYSCALE:
		out << "grayscale";
		break;
	case model::RGB:
		out << "rgb";
		break;
	case model::CMY:
		out << "cmy";
		break;
	case model::YIQ:
		out << "yiq";
		break;
	case model::HSV:
		out << "hsv";
		break;
	}

	return out;
}

}}
