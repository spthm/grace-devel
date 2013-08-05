#ifndef S3D_COLOR_COLOR_H
#define S3D_COLOR_COLOR_H

#include "cmy.h"
#include "hsv.h"
#include "radiance.h"
#include "rgb.h"
#include "yiq.h"
#include "alpha.h"

namespace s3d { namespace color {

std::ostream &operator<<(std::ostream &out, const model &s);

}}

#endif
