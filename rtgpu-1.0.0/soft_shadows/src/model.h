#ifndef RTGPU_MODEL_H
#define RTGPU_MODEL_H

#include <s3d++/mesh/face.h>

struct model
{
    std::vector<s3d::math::face<int,3>> faces;
    std::vector<s3d::r3::point> positions;
    std::vector<s3d::r3::vector> normals;
    std::vector<s3d::r2::param_coord> texcoords;
    std::vector<s3d::color::rgb> colors;
};

model load_model(const std::string &fname);

s3d::r3::box bounding_box(const model &m);

model make_torus(float R, float r, size_t dw, size_t dh);

#endif
