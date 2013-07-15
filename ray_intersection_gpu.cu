#include "cudaErrorCheck.h"
#include "ray_intersection.h"
#include <stdio.h>

__global__ void gpu_ray_cells_pluecker(cpp_src_ray_type *ray,
                                       double *s2b, double *s2t,
                                       bool *hits, int Ncells) {
  // s2b is vector from ray start to lower cell corner.
  // s2t is vector from ray start to upper cell corner.

  double *dir;
  double dist;
  double e2b[3]; // Vector from ray end to lower cell corner.
  double e2t[3]; // Vector from ray end to upper cell corner.

  int tid = blockIdx.x;// * blockDim.x + threadIdx.x;

  dir = ray->dir;
  dist = ray->length;

  if (tid < Ncells) {

    for (int i=0; i<3; i++) {
      // s2x is s2x(N,3) in Fortran, so (x1, x2, ... y1, y2, ... z1, z2, ...)
      // in memory.
      e2b[i] = s2b[tid+Ncells*i] - dir[i] * dist;
      e2t[i] = s2t[tid+Ncells*i] - dir[i] * dist;
    }

    // Assume true.  If ray misses then it will be set to false in the switch.
    hits[tid] = true;

    switch(ray->dir_class) {
    // MMM
    //-----------
    case 0:
      if(s2b[0] > 0.0 || s2b[1] > 0.0 || s2b[2] > 0.0)
        hits[tid] = false; // on negative part of ray

      else if(e2t[0] < 0.0 || e2t[1] < 0.0 || e2t[2] < 0.0)
        hits[tid] = false; // past length of ray

      else if (dir[0]*s2b[1] - dir[1]*s2t[0] < 0.0 ||
               dir[0]*s2t[1] - dir[1]*s2b[0] > 0.0 ||
               dir[0]*s2t[2] - dir[2]*s2b[0] > 0.0 ||
               dir[0]*s2b[2] - dir[2]*s2t[0] < 0.0 ||
               dir[1]*s2b[2] - dir[2]*s2t[1] < 0.0 ||
               dir[1]*s2t[2] - dir[2]*s2b[1] > 0.0)
        hits[tid] = false;
      break;

    // PMM
    //-----------
    case 1:
      if(s2t[0] < 0.0 || s2b[1] > 0.0 || s2b[2] > 0.0)
        hits[tid] = false; // on negative part of ray

      else if(e2b[0] > 0.0 || e2t[1] < 0.0 || e2t[2] < 0.0)
        hits[tid] = false; // past length of ray


      else if (dir[0]*s2t[1] - dir[1]*s2t[0] < 0.0 ||
               dir[0]*s2b[1] - dir[1]*s2b[0] > 0.0 ||
               dir[0]*s2b[2] - dir[2]*s2b[0] > 0.0 ||
               dir[0]*s2t[2] - dir[2]*s2t[0] < 0.0 ||
               dir[1]*s2b[2] - dir[2]*s2t[1] < 0.0 ||
               dir[1]*s2t[2] - dir[2]*s2b[1] > 0.0)
        hits[tid] = false;
      break;

    // MPM
    //-----------
    case 2:
      if(s2b[0] > 0.0 || s2t[1] < 0.0 || s2b[2] > 0.0)
        hits[tid] = false; // on negative part of ray

      else if(e2t[0] < 0.0 || e2b[1] > 0.0 || e2t[2] < 0.0)
        hits[tid] = false; // past length of ray

      else if (dir[0]*s2b[1] - dir[1]*s2b[0] < 0.0 ||
               dir[0]*s2t[1] - dir[1]*s2t[0] > 0.0 ||
               dir[0]*s2t[2] - dir[2]*s2b[0] > 0.0 ||
               dir[0]*s2b[2] - dir[2]*s2t[0] < 0.0 ||
               dir[1]*s2t[2] - dir[2]*s2t[1] < 0.0 ||
               dir[1]*s2b[2] - dir[2]*s2b[1] > 0.0)
        hits[tid] = false;
      break;

    // PPM
    //-----------
    case 3:
      if(s2t[0] < 0.0 || s2t[1] < 0.0 || s2b[2] > 0.0)
        hits[tid] = false; // on negative part of ray

      else if(e2b[0] > 0.0 || e2b[1] > 0.0 || e2t[2] < 0.0)
        hits[tid] = false; // past length of ray

      else if (dir[0]*s2t[1] - dir[1]*s2b[0] < 0.0 ||
               dir[0]*s2b[1] - dir[1]*s2t[0] > 0.0 ||
               dir[0]*s2b[2] - dir[2]*s2b[0] > 0.0 ||
               dir[0]*s2t[2] - dir[2]*s2t[0] < 0.0 ||
               dir[1]*s2t[2] - dir[2]*s2t[1] < 0.0 ||
               dir[1]*s2b[2] - dir[2]*s2b[1] > 0.0)
        hits[tid] = false;
      break;

    // MMP
    //-----------
    case 4:
      if(s2b[0] > 0.0 || s2b[1] > 0.0 || s2t[2] < 0.0)
        hits[tid] = false; // on negative part of ray

      else if(e2t[0] < 0.0 || e2t[1] < 0.0 || e2b[2] > 0.0)
        hits[tid] = false; // past length of ray

      else if (dir[0]*s2b[1] - dir[1]*s2t[0] < 0.0 ||
               dir[0]*s2t[1] - dir[1]*s2b[0] > 0.0 ||
               dir[0]*s2t[2] - dir[2]*s2t[0] > 0.0 ||
               dir[0]*s2b[2] - dir[2]*s2b[0] < 0.0 ||
               dir[1]*s2b[2] - dir[2]*s2b[1] < 0.0 ||
               dir[1]*s2t[2] - dir[2]*s2t[1] > 0.0)
        hits[tid] = false;
      break;

    // PMP
    //-----------
    case 5:
      if(s2t[0] < 0.0 || s2b[1] > 0.0 || s2t[2] < 0.0)
        hits[tid] = false; // on negative part of ray

      else if(e2b[0] > 0.0 || e2t[1] < 0.0 || e2b[2] > 0.0)
        hits[tid] = false; // past length of ray

      else if (dir[0]*s2t[1] - dir[1]*s2t[0] < 0.0 ||
               dir[0]*s2b[1] - dir[1]*s2b[0] > 0.0 ||
               dir[0]*s2b[2] - dir[2]*s2t[0] > 0.0 ||
               dir[0]*s2t[2] - dir[2]*s2b[0] < 0.0 ||
               dir[1]*s2b[2] - dir[2]*s2b[1] < 0.0 ||
               dir[1]*s2t[2] - dir[2]*s2t[1] > 0.0)
        hits[tid] = false;
      break;

    // MPP
    //-----------
    case 6:
      if(s2b[0] > 0.0 || s2t[1] < 0.0 || s2t[2] < 0.0)
        hits[tid] = false; // on negative part of ray

      else if(e2t[0] < 0.0 || e2b[1] > 0.0 || e2b[2] > 0.0)
        hits[tid] = false; // past length of ray

      else if (dir[0]*s2b[1] - dir[1]*s2b[0] < 0.0 ||
               dir[0]*s2t[1] - dir[1]*s2t[0] > 0.0 ||
               dir[0]*s2t[2] - dir[2]*s2t[0] > 0.0 ||
               dir[0]*s2b[2] - dir[2]*s2b[0] < 0.0 ||
               dir[1]*s2t[2] - dir[2]*s2b[1] < 0.0 ||
               dir[1]*s2b[2] - dir[2]*s2t[1] > 0.0)
        hits[tid] = false;
      break;

    // PPP
    //-----------
    case 7:
      if(s2t[0] < 0.0 || s2t[1] < 0.0 || s2t[2] < 0.0)
        hits[tid] = false; // on negative part of ray

      else if(e2b[0] > 0.0 || e2b[1] > 0.0 || e2b[2] > 0.0)
        hits[tid] = false; // past length of ray

      else if (dir[0]*s2t[1] - dir[1]*s2b[0] < 0.0 ||
               dir[0]*s2b[1] - dir[1]*s2t[0] > 0.0 ||
               dir[0]*s2b[2] - dir[2]*s2t[0] > 0.0 ||
               dir[0]*s2t[2] - dir[2]*s2b[0] < 0.0 ||
               dir[1]*s2t[2] - dir[2]*s2b[1] < 0.0 ||
               dir[1]*s2b[2] - dir[2]*s2t[1] > 0.0)
        hits[tid] = false;
      break;
      }
    }
}

extern "C" void cu_src_ray_pluecker_(cpp_src_ray_type *src_ray,
                                     double *s2b, double *s2t,
                                     int *Ncells, bool *hits) {
  int N = *Ncells;
  bool *dev_hits;
  cpp_src_ray_type *dev_ray;
  double *dev_s2b;
  double *dev_s2t;

  CUDA_HANDLE_ERR( cudaMalloc( (void**)&dev_ray, sizeof(*src_ray) ) );
  CUDA_HANDLE_ERR( cudaMalloc( (void**)&dev_s2b, 3*N*sizeof(*s2b) ) );
  CUDA_HANDLE_ERR( cudaMalloc( (void**)&dev_s2t, 3*N*sizeof(*s2t) ) );
  CUDA_HANDLE_ERR( cudaMalloc( (void**)&dev_hits, N*sizeof(*hits) ) );

  CUDA_HANDLE_ERR( cudaMemcpy(dev_ray, src_ray, sizeof(*src_ray),
                              cudaMemcpyHostToDevice) );
  CUDA_HANDLE_ERR( cudaMemcpy(dev_s2b, s2b, 3*N*sizeof(*s2b),
                              cudaMemcpyHostToDevice) );
  CUDA_HANDLE_ERR( cudaMemcpy(dev_s2t, s2t, 3*N*sizeof(*s2t),
                              cudaMemcpyHostToDevice) );

  gpu_ray_cells_pluecker<<<N,1>>>(dev_ray, dev_s2b, dev_s2t,
                                          dev_hits, N);

  CUDA_HANDLE_ERR( cudaMemcpy(hits, dev_hits, N*sizeof(*hits),
                              cudaMemcpyDeviceToHost) );

  cudaFree(dev_ray);
  cudaFree(dev_s2b);
  cudaFree(dev_s2t);
  cudaFree(dev_hits);

}

__global__ void gpu_ray_cell_slope(slope_ray_type *r, double *bot, double *top,
                              bool *hit) {

  *hit = true;

  switch (r->classification)
  {
    case MMM:
      if ((r->x < bot[0]) || (r->y < bot[1]) || (r->z < bot[2])
        || (r->jbyi * bot[0] - top[1] + r->c_xy > 0)
        || (r->ibyj * bot[1] - top[0] + r->c_yx > 0)
        || (r->jbyk * bot[2] - top[1] + r->c_zy > 0)
        || (r->kbyj * bot[1] - top[2] + r->c_yz > 0)
        || (r->kbyi * bot[0] - top[2] + r->c_xz > 0)
        || (r->ibyk * bot[2] - top[0] + r->c_zx > 0)
        )
        *hit = false;
      break;

    case MMP:

      if ((r->x < bot[0]) || (r->y < bot[1]) || (r->z > top[2])
        || (r->jbyi * bot[0] - top[1] + r->c_xy > 0)
        || (r->ibyj * bot[1] - top[0] + r->c_yx > 0)
        || (r->jbyk * top[2] - top[1] + r->c_zy > 0)
        || (r->kbyj * bot[1] - bot[2] + r->c_yz < 0)
        || (r->kbyi * bot[0] - bot[2] + r->c_xz < 0)
        || (r->ibyk * top[2] - top[0] + r->c_zx > 0)
        )
        *hit = false;
      break;

    case MPM:

      if ((r->x < bot[0]) || (r->y > top[1]) || (r->z < bot[2])
        || (r->jbyi * bot[0] - bot[1] + r->c_xy < 0)
        || (r->ibyj * top[1] - top[0] + r->c_yx > 0)
        || (r->jbyk * bot[2] - bot[1] + r->c_zy < 0)
        || (r->kbyj * top[1] - top[2] + r->c_yz > 0)
        || (r->kbyi * bot[0] - top[2] + r->c_xz > 0)
        || (r->ibyk * bot[2] - top[0] + r->c_zx > 0)
        )
        *hit = false;
      break;

    case MPP:

      if ((r->x < bot[0]) || (r->y > top[1]) || (r->z > top[2])
        || (r->jbyi * bot[0] - bot[1] + r->c_xy < 0)
        || (r->ibyj * top[1] - top[0] + r->c_yx > 0)
        || (r->jbyk * top[2] - bot[1] + r->c_zy < 0)
        || (r->kbyj * top[1] - bot[2] + r->c_yz < 0)
        || (r->kbyi * bot[0] - bot[2] + r->c_xz < 0)
        || (r->ibyk * top[2] - top[0] + r->c_zx > 0)
        )
        *hit = false;
      break;

    case PMM:

      if ((r->x > top[0]) || (r->y < bot[1]) || (r->z < bot[2])
        || (r->jbyi * top[0] - top[1] + r->c_xy > 0)
        || (r->ibyj * bot[1] - bot[0] + r->c_yx < 0)
        || (r->jbyk * bot[2] - top[1] + r->c_zy > 0)
        || (r->kbyj * bot[1] - top[2] + r->c_yz > 0)
        || (r->kbyi * top[0] - top[2] + r->c_xz > 0)
        || (r->ibyk * bot[2] - bot[0] + r->c_zx < 0)
        )
        *hit = false;
      break;

    case PMP:

      if ((r->x > top[0]) || (r->y < bot[1]) || (r->z > top[2])
        || (r->jbyi * top[0] - top[1] + r->c_xy > 0)
        || (r->ibyj * bot[1] - bot[0] + r->c_yx < 0)
        || (r->jbyk * top[2] - top[1] + r->c_zy > 0)
        || (r->kbyj * bot[1] - bot[2] + r->c_yz < 0)
        || (r->kbyi * top[0] - bot[2] + r->c_xz < 0)
        || (r->ibyk * top[2] - bot[0] + r->c_zx < 0)
        )
        *hit = false;
      break;

    case PPM:

      if ((r->x > top[0]) || (r->y > top[1]) || (r->z < bot[2])
        || (r->jbyi * top[0] - bot[1] + r->c_xy < 0)
        || (r->ibyj * top[1] - bot[0] + r->c_yx < 0)
        || (r->jbyk * bot[2] - bot[1] + r->c_zy < 0)
        || (r->kbyj * top[1] - top[2] + r->c_yz > 0)
        || (r->kbyi * top[0] - top[2] + r->c_xz > 0)
        || (r->ibyk * bot[2] - bot[0] + r->c_zx < 0)
        )
        *hit = false;
      break;

    case PPP:

      if ((r->x > top[0]) || (r->y > top[1]) || (r->z > top[2])
        || (r->jbyi * top[0] - bot[1] + r->c_xy < 0)
        || (r->ibyj * top[1] - bot[0] + r->c_yx < 0)
        || (r->jbyk * top[2] - bot[1] + r->c_zy < 0)
        || (r->kbyj * top[1] - bot[2] + r->c_yz < 0)
        || (r->kbyi * top[0] - bot[2] + r->c_xz < 0)
        || (r->ibyk * top[2] - bot[0] + r->c_zx < 0)
        )
        *hit = false;
      break;

    case OMM:

      if((r->x < bot[0]) || (r->x > top[0])
        || (r->y < bot[1]) || (r->z < bot[2])
        || (r->jbyk * bot[2] - top[1] + r->c_zy > 0)
        || (r->kbyj * bot[1] - top[2] + r->c_yz > 0)
        )
        *hit = false;
      break;

    case OMP:

      if((r->x < bot[0]) || (r->x > top[0])
        || (r->y < bot[1]) || (r->z > top[2])
        || (r->jbyk * top[2] - top[1] + r->c_zy > 0)
        || (r->kbyj * bot[1] - bot[2] + r->c_yz < 0)
        )
        *hit = false;
      break;

    case OPM:

      if((r->x < bot[0]) || (r->x > top[0])
        || (r->y > top[1]) || (r->z < bot[2])
        || (r->jbyk * bot[2] - bot[1] + r->c_zy < 0)
        || (r->kbyj * top[1] - top[2] + r->c_yz > 0)
        )
        *hit = false;
      break;

    case OPP:

      if((r->x < bot[0]) || (r->x > top[0])
        || (r->y > top[1]) || (r->z > top[2])
        || (r->jbyk * top[2] - bot[1] + r->c_zy < 0)
        || (r->kbyj * top[1] - bot[2] + r->c_yz < 0)
        )
        *hit = false;
      break;

    case MOM:

      if((r->y < bot[1]) || (r->y > top[1])
        || (r->x < bot[0]) || (r->z < bot[2])
        || (r->kbyi * bot[0] - top[2] + r->c_xz > 0)
        || (r->ibyk * bot[2] - top[0] + r->c_zx > 0)
        )
        *hit = false;
      break;

    case MOP:

      if((r->y < bot[1]) || (r->y > top[1])
        || (r->x < bot[0]) || (r->z > top[2])
        || (r->kbyi * bot[0] - bot[2] + r->c_xz < 0)
        || (r->ibyk * top[2] - top[0] + r->c_zx > 0)
        )
        *hit = false;
      break;

    case POM:

      if((r->y < bot[1]) || (r->y > top[1])
        || (r->x > top[0]) || (r->z < bot[2])
        || (r->kbyi * top[0] - top[2] + r->c_xz > 0)
        || (r->ibyk * bot[2] - bot[0] + r->c_zx < 0)
        )
        *hit = false;
      break;

    case POP:

      if((r->y < bot[1]) || (r->y > top[1])
        || (r->x > top[0]) || (r->z > top[2])
        || (r->kbyi * top[0] - bot[2] + r->c_xz < 0)
        || (r->ibyk * top[2] - bot[0] + r->c_zx < 0)
        )
        *hit = false;
      break;

    case MMO:

      if((r->z < bot[2]) || (r->z > top[2])
        || (r->x < bot[0]) || (r->y < bot[1])
        || (r->jbyi * bot[0] - top[1] + r->c_xy > 0)
        || (r->ibyj * bot[1] - top[0] + r->c_yx > 0)
        )
        *hit = false;
      break;

    case MPO:

      if((r->z < bot[2]) || (r->z > top[2])
        || (r->x < bot[0]) || (r->y > top[1])
        || (r->jbyi * bot[0] - bot[1] + r->c_xy < 0)
        || (r->ibyj * top[1] - top[0] + r->c_yx > 0)
        )
        *hit = false;
      break;

    case PMO:

      if((r->z < bot[2]) || (r->z > top[2])
        || (r->x > top[0]) || (r->y < bot[1])
        || (r->jbyi * top[0] - top[1] + r->c_xy > 0)
        || (r->ibyj * bot[1] - bot[0] + r->c_yx < 0)
        )
        *hit = false;
      break;

    case PPO:

      if((r->z < bot[2]) || (r->z > top[2])
        || (r->x > top[0]) || (r->y > top[1])
        || (r->jbyi * top[0] - bot[1] + r->c_xy < 0)
        || (r->ibyj * top[1] - bot[0] + r->c_yx < 0)
        )
        *hit = false;
      break;

    case MOO:

      if((r->x < bot[0])
        || (r->y < bot[1]) || (r->y > top[1])
        || (r->z < bot[2]) || (r->z > top[2])
        )
        *hit = false;
      break;

    case POO:

      if((r->x > top[0])
        || (r->y < bot[1]) || (r->y > top[1])
        || (r->z < bot[2]) || (r->z > top[2])
        )
        *hit = false;
      break;

    case OMO:

      if((r->y < bot[1])
        || (r->x < bot[0]) || (r->x > top[0])
        || (r->z < bot[2]) || (r->z > top[2])
        )
        *hit = false;

    // Deliberate fall-through!?
    case OPO:

      if((r->y > top[1])
        || (r->x < bot[0]) || (r->x > top[0])
        || (r->z < bot[2]) || (r->z > top[2])
        )
        *hit = false;

    case OOM:

      if((r->z < bot[2])
        || (r->x < bot[0]) || (r->x > top[0])
        || (r->y < bot[1]) || (r->y > top[1])
        )
        *hit = false;

    case OOP:

      if((r->z > top[2])
        || (r->x < bot[0]) || (r->x > top[0])
        || (r->y < bot[1]) || (r->y > top[1])
        )
        *hit = false;
      break;
  }
}

extern "C" bool cu_ray_slope_(slope_ray_type *ray, double *bot, double *top) {
  bool hit;
  bool *dev_hit;
  slope_ray_type *dev_ray;
  double *dev_bot;
  double *dev_top;

  cudaMalloc( (void**)&dev_ray, sizeof(*ray) );
  cudaMalloc( (void**)&dev_bot, 3 * sizeof(*bot) );
  cudaMalloc( (void**)&dev_top, 3 * sizeof(*top) );
  cudaMalloc( (void**)&dev_hit, sizeof(hit) );

  cudaMemcpy(dev_ray, ray, sizeof(*ray), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_bot, bot, 3 * sizeof(*bot), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_top, top, 3 * sizeof(*top), cudaMemcpyHostToDevice);

  gpu_ray_cell_slope<<<1,1>>>(dev_ray, dev_bot, dev_top, dev_hit);

  cudaMemcpy(&hit, dev_hit, sizeof(hit), cudaMemcpyDeviceToHost);

  cudaFree(dev_ray);
  cudaFree(dev_bot);
  cudaFree(dev_top);
  cudaFree(dev_hit);

  return hit;
}
