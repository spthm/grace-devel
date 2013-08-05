#include <optix.h>
#include <optix_math.h>
#include <optixu/optixu_matrix.h>
#include <optixu/optixu_aabb.h>

// This is to be plugged into an RTgeometry object to represent
// a triangle mesh with a vertex buffer of triangle soup (triangle list)
// with an interleaved position, normal, texturecoordinate layout.

rtBuffer<float3> vertex_buffer;     
rtBuffer<float3> normal_buffer;
//rtBuffer<float2> texcoord_buffer;

rtBuffer<int3>   index_buffer;    // position indices 

//rtBuffer<int3>   vindex_buffer;    // position indices 
//rtBuffer<int3>   nindex_buffer;    // normal indices
//rtBuffer<int3>   tindex_buffer;    // texcoord indices

//rtBuffer<uint>   material_buffer; // per-face material index
//rtDeclareVariable(float3, texcoord, attribute texcoord, ); 
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, ); 
rtDeclareVariable(float3, shading_normal, attribute shading_normal, ); 
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );

RT_PROGRAM void intersect( int primIdx )
{
  int3 v_idx = index_buffer[primIdx];

  float3 p0 = vertex_buffer[ v_idx.x ];
  float3 p1 = vertex_buffer[ v_idx.y ];
  float3 p2 = vertex_buffer[ v_idx.z ];

  // Intersect ray with triangle
  float3 e0 = p1 - p0;
  float3 e1 = p0 - p2;
  float3 n  = cross( e0, e1 );

  float v   = dot( n, ray.direction );
  float r   = 1.0f / v;

  float3 e2 = p0 - ray.origin;
  float va  = dot( n, e2 );
  float t   = r*va;

  if(t < ray.tmax && t > ray.tmin) {
    float3 i   = cross( e2, ray.direction );
    float v1   = dot( i, e1 );
    float beta = r*v1;
    if(beta >= 0.0f){
      float v2 = dot( i, e0 );
      float gamma = r*v2;
      if( (v1+v2)*v <= v*v && gamma >= 0.0f ) {

        if(  rtPotentialIntersection( t ) ) {

          int3 n_idx = index_buffer[ primIdx ];

          if ( normal_buffer.size() == 0 || n_idx.x < 0 || n_idx.y < 0 || n_idx.z < 0 ) {
            shading_normal = -n;
          } else {
            float3 n0 = normal_buffer[ n_idx.x ];
            float3 n1 = normal_buffer[ n_idx.y ];
            float3 n2 = normal_buffer[ n_idx.z ];
            shading_normal = normalize( n1*beta + n2*gamma + n0*(1.0f-beta-gamma) );
          }
          geometric_normal = -n;

#if 0
          int3 t_idx = tindex_buffer[ primIdx ];
          if ( texcoord_buffer.size() == 0 || t_idx.x < 0 || t_idx.y < 0 || t_idx.z < 0 ) {
            texcoord = make_float3( 0.0f, 0.0f, 0.0f );
          } else {
            float2 t0 = texcoord_buffer[ t_idx.x ];
            float2 t1 = texcoord_buffer[ t_idx.y ];
            float2 t2 = texcoord_buffer[ t_idx.z ];
            texcoord = make_float3( t1*beta + t2*gamma + t0*(1.0f-beta-gamma) );
          }
#endif

          rtReportIntersection(0);//material_buffer[primIdx]);
        }
      }
    }
  }
}

RT_PROGRAM void bounds (int primIdx, float result[6])
{
  int3 v_idx = index_buffer[primIdx];

  float3 v0 = vertex_buffer[ v_idx.x ];
  float3 v1 = vertex_buffer[ v_idx.y ];
  float3 v2 = vertex_buffer[ v_idx.z ];

  optix::Aabb* aabb = (optix::Aabb*)result;
  aabb->m_min = fminf( fminf( v0, v1), v2 );
  aabb->m_max = fmaxf( fmaxf( v0, v1), v2 );
}
