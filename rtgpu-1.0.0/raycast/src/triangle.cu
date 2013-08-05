#include "types.h"

inline 
__device__ bool intersect(float3 o, float3 d, 
                          float tmin, float tmax,
                          const texture<float4> &xform,
                          const texture<float4> &normals,
                          unsigned idtri,
                          float *t_ray=NULL, float3 *shading_normal=NULL,
                          float3 *geom_normal=NULL)
{
    float4 mp0 = tex1Dfetch(xform, idtri+0), 
           mp1 = tex1Dfetch(xform, idtri+1),
           mp2 = tex1Dfetch(xform, idtri+2);

    float3 to, td;

    to.z = mp2.x*o.x + mp2.y*o.y + mp2.z*o.z + mp2.w;
    td.z = mp2.x*d.x + mp2.y*d.y + mp2.z*d.z;

    float t = __fdividef(-to.z,td.z);

    if(t > tmin && t < tmax)
    {
        to.x = mp0.x*o.x + mp0.y*o.y + mp0.z*o.z + mp0.w;
        td.x = mp0.x*d.x + mp0.y*d.y + mp0.z*d.z;

        float u = t*td.x + to.x;

        if(u >= 0)
        {
            to.y = mp1.x*o.x + mp1.y*o.y + mp1.z*o.z + mp1.w;
            td.y = mp1.x*d.x + mp1.y*d.y + mp1.z*d.z;

            float v = t*td.y + to.y;
            if(v >= 0 && u+v <= 1)
            {
                if(t_ray)
                    *t_ray = t;

                if(shading_normal)
                {
                    float4 n0 = tex1Dfetch(normals, idtri+0), 
                           n1 = tex1Dfetch(normals, idtri+1),
                           n2 = tex1Dfetch(normals, idtri+2);
                    *shading_normal = make_float3(n0.x,n0.y,n0.z)*u + 
                                      make_float3(n1.x,n1.y,n1.z)*v + 
                                      make_float3(n2.x,n2.y,n2.z)*(1.0f-u-v);
                }
                return true;
            }
        }
    }

    return false;

#if 0

    float3 e0 = p1 - p0;
    float3 e1 = p0 - p2;
    float3 e2 = p0 - ray_ori;

    float3 n  = cross(e0, e1);

    float v = dot(n, ray_dir);
    float va = dot(n,e2);
          
    float r = 1.0f/v;

    float t = r*va;

    if(t > ray_tmin && t < ray_tmax)
    {
        float3 i = cross(e2, ray_dir);
        float v1 = dot(i,e1);
        float beta = r*v1;
        if(beta >= 0)
        {
            float v2 = dot(i,e0);
            float gamma = r*v2;
            if((v1+v2)*v <= v*v && gamma >= 0)
            {
                if(t_ray)
                    *t_ray = t;

                if(geom_normal)
                    *geom_normal = -n;

                if(shading_normal)
                {
                    float3 n0 = normals[idx.x];
                    float3 n1 = normals[idx.y];
                    float3 n2 = normals[idx.z];
                    *shading_normal = n1*beta + n2*gamma + n0*(1.0f-beta-gamma);
                }

                return true;
            }
        }
    }

    return false;
#endif
                

#if 0
    // Ref: Fast, Minimum Storage Ray/Triangle Intersection
    //      Tomas Möller and Ben Trumbore

    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;

    float3 pvec = cross(ray_dir,e2);

    float det = dot(e1, pvec);
    if(fabs(det) < 1e-6f)
        return false;

    float inv_det = 1.0f/det;

    float3 tvec = ray_ori - p0;

    float u = dot(tvec,pvec) * inv_det;
    if(u < 0 || u > 1)
        return false;

    float3 qvec = cross(tvec, e1);
    float v = dot(ray_dir, qvec) * inv_det;

    if(v < 0 || u+v > 1)
        return false;

    float t = dot(e2,qvec) * inv_det; 

    if(t >= ray_tmin && t < ray_tmax)
    {
        if(shading_normal)
        {
            float3 n0 = normals[idx.x];
            float3 n1 = normals[idx.y];
            float3 n2 = normals[idx.z];
            *shading_normal = n1*u + n2*v + n0*(1.0f-u-v);
        }

        if(geom_normal)
            *geom_normal = cross(e1,e2);

        if(t_ray)
            *t_ray = t;

        return true;
    }

    return false;
#endif
}
