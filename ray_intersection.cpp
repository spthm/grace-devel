#include <string.h>
#include <stdio.h>
#include "ray_intersection.h"

/*void rayError_(char *string, char *len);*/

extern "C" bool cpp_src_ray_pluecker_(cpp_src_ray_type *ray, double *s2b,
                                      double *s2t, int *reason) {
  double zero = 0.0;

  double *dir;
  double dist;
  double e2b[3]; // Vector from ray end to lower cell corner.
  double e2t[3]; // Vector from ray end to upper cell corner.

  dir = ray->dir;
  dist = (*ray).length;

  for (int i=0; i<3; i++) {
  e2b[i] = s2b[i] - dir[i] * dist;
  e2t[i] = s2t[i] - dir[i] * dist;
  }

  // Assume true.  If ray misses then it will be set to false in the switch.
  bool hit = true;

  switch(ray->dir_class) {
  // MMM
  //-----------
  case 0:
    if(s2b[0] > zero || s2b[1] > zero || s2b[2] > zero)
      hit = false; // on negative part of ray

    else if(e2t[0] < zero || e2t[1] < zero || e2t[2] < zero)
      hit = false; // past length of ray

    else if (dir[0]*s2b[1] - dir[1]*s2t[0] < zero ||
             dir[0]*s2t[1] - dir[1]*s2b[0] > zero ||
             dir[0]*s2t[2] - dir[2]*s2b[0] > zero ||
             dir[0]*s2b[2] - dir[2]*s2t[0] < zero ||
             dir[1]*s2b[2] - dir[2]*s2t[1] < zero ||
             dir[1]*s2t[2] - dir[2]*s2b[1] > zero)
      hit = false;
    break;

  // PMM
  //-----------
  case 1:
    if(s2t[0] < zero || s2b[1] > zero || s2b[2] > zero)
      hit = false; // on negative part of ray

    else if(e2b[0] > zero || e2t[1] < zero || e2t[2] < zero)
      hit = false; // past length of ray


    else if (dir[0]*s2t[1] - dir[1]*s2t[0] < zero ||
             dir[0]*s2b[1] - dir[1]*s2b[0] > zero ||
             dir[0]*s2b[2] - dir[2]*s2b[0] > zero ||
             dir[0]*s2t[2] - dir[2]*s2t[0] < zero ||
             dir[1]*s2b[2] - dir[2]*s2t[1] < zero ||
             dir[1]*s2t[2] - dir[2]*s2b[1] > zero)
      hit = false;
    break;

  // MPM
  //-----------
  case 2:
    if(s2b[0] > zero || s2t[1] < zero || s2b[2] > zero)
      hit = false; // on negative part of ray

    else if(e2t[0] < zero || e2b[1] > zero || e2t[2] < zero)
      hit = false; // past length of ray

    else if (dir[0]*s2b[1] - dir[1]*s2b[0] < zero ||
             dir[0]*s2t[1] - dir[1]*s2t[0] > zero ||
             dir[0]*s2t[2] - dir[2]*s2b[0] > zero ||
             dir[0]*s2b[2] - dir[2]*s2t[0] < zero ||
             dir[1]*s2t[2] - dir[2]*s2t[1] < zero ||
             dir[1]*s2b[2] - dir[2]*s2b[1] > zero)
      hit = false;
    break;

  // PPM
  //-----------
  case 3:
    if(s2t[0] < zero || s2t[1] < zero || s2b[2] > zero)
      hit = false; // on negative part of ray

    else if(e2b[0] > zero || e2b[1] > zero || e2t[2] < zero)
      hit = false; // past length of ray

    else if (dir[0]*s2t[1] - dir[1]*s2b[0] < zero ||
             dir[0]*s2b[1] - dir[1]*s2t[0] > zero ||
             dir[0]*s2b[2] - dir[2]*s2b[0] > zero ||
             dir[0]*s2t[2] - dir[2]*s2t[0] < zero ||
             dir[1]*s2t[2] - dir[2]*s2t[1] < zero ||
             dir[1]*s2b[2] - dir[2]*s2b[1] > zero)
      hit = false;
    break;

  // MMP
  //-----------
  case 4:
    if(s2b[0] > zero || s2b[1] > zero || s2t[2] < zero)
      hit = false; // on negative part of ray

    else if(e2t[0] < zero || e2t[1] < zero || e2b[2] > zero)
      hit = false; // past length of ray

    else if (dir[0]*s2b[1] - dir[1]*s2t[0] < zero ||
             dir[0]*s2t[1] - dir[1]*s2b[0] > zero ||
             dir[0]*s2t[2] - dir[2]*s2t[0] > zero ||
             dir[0]*s2b[2] - dir[2]*s2b[0] < zero ||
             dir[1]*s2b[2] - dir[2]*s2b[1] < zero ||
             dir[1]*s2t[2] - dir[2]*s2t[1] > zero)
      hit = false;
    break;

  // PMP
  //-----------
  case 5:
    if(s2t[0] < zero || s2b[1] > zero || s2t[2] < zero)
      hit = false; // on negative part of ray

    else if(e2b[0] > zero || e2t[1] < zero || e2b[2] > zero)
      hit = false; // past length of ray

    else if (dir[0]*s2t[1] - dir[1]*s2t[0] < zero ||
             dir[0]*s2b[1] - dir[1]*s2b[0] > zero ||
             dir[0]*s2b[2] - dir[2]*s2t[0] > zero ||
             dir[0]*s2t[2] - dir[2]*s2b[0] < zero ||
             dir[1]*s2b[2] - dir[2]*s2b[1] < zero ||
             dir[1]*s2t[2] - dir[2]*s2t[1] > zero)
      hit = false;
    break;

  // MPP
  //-----------
  case 6:
    if(s2b[0] > zero || s2t[1] < zero || s2t[2] < zero)
      hit = false; // on negative part of ray

    else if(e2t[0] < zero || e2b[1] > zero || e2b[2] > zero)
      hit = false; // past length of ray

    else if (dir[0]*s2b[1] - dir[1]*s2b[0] < zero ||
             dir[0]*s2t[1] - dir[1]*s2t[0] > zero ||
             dir[0]*s2t[2] - dir[2]*s2t[0] > zero ||
             dir[0]*s2b[2] - dir[2]*s2b[0] < zero ||
             dir[1]*s2t[2] - dir[2]*s2b[1] < zero ||
             dir[1]*s2b[2] - dir[2]*s2t[1] > zero)
      hit = false;
    break;

  // PPP
  //-----------
  case 7:
    if(s2t[0] < zero || s2t[1] < zero || s2t[2] < zero)
      hit = false; // on negative part of ray

    else if(e2b[0] > zero || e2b[1] > zero || e2b[2] > zero)
      hit = false; // past length of ray

    else if (dir[0]*s2t[1] - dir[1]*s2b[0] < zero ||
             dir[0]*s2b[1] - dir[1]*s2t[0] > zero ||
             dir[0]*s2b[2] - dir[2]*s2t[0] > zero ||
             dir[0]*s2t[2] - dir[2]*s2b[0] < zero ||
             dir[1]*s2t[2] - dir[2]*s2b[1] < zero ||
             dir[1]*s2b[2] - dir[2]*s2t[1] > zero)
      hit = false;
    break;

  }

  return hit;
}
