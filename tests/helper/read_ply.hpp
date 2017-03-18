#pragma once

#include "grace/vector.h"

#include <string>
#include <vector>

struct PLYTriangle {
    grace::Vector<3, float> v1, v2, v3;
};

// A return value of 0 implies success, else a failure.
int read_triangles(const std::string& ply_fname,
                   std::vector<PLYTriangle>& tris);
