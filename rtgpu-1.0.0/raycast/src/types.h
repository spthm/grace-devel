#ifndef TYPES_H
#define TYPES_H
 
#include <optix_math.h>

#define ROBUST 0
const float RAY_EPSILON = 0.001;

#define MAX_LIGHTS 1
#define MAX_SPHERES 5

struct Sphere
{
#ifndef __CUDACC__
    Sphere(float3 c, float r) : center(c), radius(r) {}
#endif

    float3 center;
    float radius;
};

struct Mesh
{
    std::vector<float3> normals;

    // affine transform into unit triangle
    std::vector<float4> xform; 
};

#endif
