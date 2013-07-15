#include "cudaErrorCheck.h"
#include "ray_intersection_float.h"

__global__ void gpu_ray_cell_pluecker(cpp_src_ray_type *ray,
                                      float *s2b, float *s2t, bool *hit) {
// s2b is vector from ray start to lower cell corner.
// s2t is vector from ray start to upper cell corner.

float *dir;
float dist;
float e2b[3]; // Vector from ray end to lower cell corner.
float e2t[3]; // Vector from ray end to upper cell corner.

dir = ray->dir;
dist = (*ray).length;

for (int i=0; i<3; i++) {
  e2b[i] = s2b[i] - dir[i] * dist;
  e2t[i] = s2t[i] - dir[i] * dist;
}

// Assume true.  If ray misses then it will be set to false in the switch.
*hit = true;

switch(ray->dir_class) {
// MMM
//-----------
case 0:
  if(s2b[0] > 0.0 || s2b[1] > 0.0 || s2b[2] > 0.0)
    *hit = false; // on negative part of ray

  else if(e2t[0] < 0.0 || e2t[1] < 0.0 || e2t[2] < 0.0)
    *hit = false; // past length of ray

  else if (dir[0]*s2b[1] - dir[1]*s2t[0] < 0.0 ||
           dir[0]*s2t[1] - dir[1]*s2b[0] > 0.0 ||
           dir[0]*s2t[2] - dir[2]*s2b[0] > 0.0 ||
           dir[0]*s2b[2] - dir[2]*s2t[0] < 0.0 ||
           dir[1]*s2b[2] - dir[2]*s2t[1] < 0.0 ||
           dir[1]*s2t[2] - dir[2]*s2b[1] > 0.0)
    *hit = false;
  break;

// PMM
//-----------
case 1:
  if(s2t[0] < 0.0 || s2b[1] > 0.0 || s2b[2] > 0.0)
    *hit = false; // on negative part of ray

  else if(e2b[0] > 0.0 || e2t[1] < 0.0 || e2t[2] < 0.0)
    *hit = false; // past length of ray


  else if (dir[0]*s2t[1] - dir[1]*s2t[0] < 0.0 ||
           dir[0]*s2b[1] - dir[1]*s2b[0] > 0.0 ||
           dir[0]*s2b[2] - dir[2]*s2b[0] > 0.0 ||
           dir[0]*s2t[2] - dir[2]*s2t[0] < 0.0 ||
           dir[1]*s2b[2] - dir[2]*s2t[1] < 0.0 ||
           dir[1]*s2t[2] - dir[2]*s2b[1] > 0.0)
    *hit = false;
  break;

// MPM
//-----------
case 2:
  if(s2b[0] > 0.0 || s2t[1] < 0.0 || s2b[2] > 0.0)
    *hit = false; // on negative part of ray

  else if(e2t[0] < 0.0 || e2b[1] > 0.0 || e2t[2] < 0.0)
    *hit = false; // past length of ray

  else if (dir[0]*s2b[1] - dir[1]*s2b[0] < 0.0 ||
           dir[0]*s2t[1] - dir[1]*s2t[0] > 0.0 ||
           dir[0]*s2t[2] - dir[2]*s2b[0] > 0.0 ||
           dir[0]*s2b[2] - dir[2]*s2t[0] < 0.0 ||
           dir[1]*s2t[2] - dir[2]*s2t[1] < 0.0 ||
           dir[1]*s2b[2] - dir[2]*s2b[1] > 0.0)
    *hit = false;
  break;

// PPM
//-----------
case 3:
  if(s2t[0] < 0.0 || s2t[1] < 0.0 || s2b[2] > 0.0)
    *hit = false; // on negative part of ray

  else if(e2b[0] > 0.0 || e2b[1] > 0.0 || e2t[2] < 0.0)
    *hit = false; // past length of ray

  else if (dir[0]*s2t[1] - dir[1]*s2b[0] < 0.0 ||
           dir[0]*s2b[1] - dir[1]*s2t[0] > 0.0 ||
           dir[0]*s2b[2] - dir[2]*s2b[0] > 0.0 ||
           dir[0]*s2t[2] - dir[2]*s2t[0] < 0.0 ||
           dir[1]*s2t[2] - dir[2]*s2t[1] < 0.0 ||
           dir[1]*s2b[2] - dir[2]*s2b[1] > 0.0)
    *hit = false;
  break;

// MMP
//-----------
case 4:
  if(s2b[0] > 0.0 || s2b[1] > 0.0 || s2t[2] < 0.0)
    *hit = false; // on negative part of ray

  else if(e2t[0] < 0.0 || e2t[1] < 0.0 || e2b[2] > 0.0)
    *hit = false; // past length of ray

  else if (dir[0]*s2b[1] - dir[1]*s2t[0] < 0.0 ||
           dir[0]*s2t[1] - dir[1]*s2b[0] > 0.0 ||
           dir[0]*s2t[2] - dir[2]*s2t[0] > 0.0 ||
           dir[0]*s2b[2] - dir[2]*s2b[0] < 0.0 ||
           dir[1]*s2b[2] - dir[2]*s2b[1] < 0.0 ||
           dir[1]*s2t[2] - dir[2]*s2t[1] > 0.0)
    *hit = false;
  break;

// PMP
//-----------
case 5:
  if(s2t[0] < 0.0 || s2b[1] > 0.0 || s2t[2] < 0.0)
    *hit = false; // on negative part of ray

  else if(e2b[0] > 0.0 || e2t[1] < 0.0 || e2b[2] > 0.0)
    *hit = false; // past length of ray

  else if (dir[0]*s2t[1] - dir[1]*s2t[0] < 0.0 ||
           dir[0]*s2b[1] - dir[1]*s2b[0] > 0.0 ||
           dir[0]*s2b[2] - dir[2]*s2t[0] > 0.0 ||
           dir[0]*s2t[2] - dir[2]*s2b[0] < 0.0 ||
           dir[1]*s2b[2] - dir[2]*s2b[1] < 0.0 ||
           dir[1]*s2t[2] - dir[2]*s2t[1] > 0.0)
    *hit = false;
  break;

// MPP
//-----------
case 6:
  if(s2b[0] > 0.0 || s2t[1] < 0.0 || s2t[2] < 0.0)
    *hit = false; // on negative part of ray

  else if(e2t[0] < 0.0 || e2b[1] > 0.0 || e2b[2] > 0.0)
    *hit = false; // past length of ray

  else if (dir[0]*s2b[1] - dir[1]*s2b[0] < 0.0 ||
           dir[0]*s2t[1] - dir[1]*s2t[0] > 0.0 ||
           dir[0]*s2t[2] - dir[2]*s2t[0] > 0.0 ||
           dir[0]*s2b[2] - dir[2]*s2b[0] < 0.0 ||
           dir[1]*s2t[2] - dir[2]*s2b[1] < 0.0 ||
           dir[1]*s2b[2] - dir[2]*s2t[1] > 0.0)
    *hit = false;
  break;

// PPP
//-----------
case 7:
  if(s2t[0] < 0.0 || s2t[1] < 0.0 || s2t[2] < 0.0)
    *hit = false; // on negative part of ray

  else if(e2b[0] > 0.0 || e2b[1] > 0.0 || e2b[2] > 0.0)
    *hit = false; // past length of ray

  else if (dir[0]*s2t[1] - dir[1]*s2b[0] < 0.0 ||
           dir[0]*s2b[1] - dir[1]*s2t[0] > 0.0 ||
           dir[0]*s2b[2] - dir[2]*s2t[0] > 0.0 ||
           dir[0]*s2t[2] - dir[2]*s2b[0] < 0.0 ||
           dir[1]*s2t[2] - dir[2]*s2b[1] < 0.0 ||
           dir[1]*s2b[2] - dir[2]*s2t[1] > 0.0)
    *hit = false;
  break;
  }
}

extern "C" bool cu_src_ray_pluecker_float_(cpp_src_ray_type *src_ray,
                                     float *s2b, float *s2t) {
  bool hit;
  bool *dev_hit;
  cpp_src_ray_type *dev_ray;
  float *dev_s2b;
  float *dev_s2t;

  cudaMalloc( (void**)&dev_ray, sizeof(*src_ray) );
  cudaMalloc( (void**)&dev_s2b, 3*sizeof(*s2b) );
  cudaMalloc( (void**)&dev_s2t, 3*sizeof(*s2t) );
  cudaMalloc( (void**)&dev_hit, sizeof(hit) );

  cudaMemcpy(dev_ray, src_ray, sizeof(*src_ray), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_s2b, s2b, 3*sizeof(*s2b), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_s2t, s2t, 3*sizeof(*s2t), cudaMemcpyHostToDevice);

  gpu_ray_cell_pluecker<<<1,1>>>(dev_ray, dev_s2b, dev_s2t, dev_hit);

  cudaMemcpy(&hit, dev_hit, sizeof(hit), cudaMemcpyDeviceToHost);

  cudaFree(dev_ray);
  cudaFree(dev_s2b);
  cudaFree(dev_s2t);
  cudaFree(dev_hit);

  return hit;
}
