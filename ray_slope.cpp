/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray / Axis-Aligned Bounding Box Overlap Tests using Ray Slopes"
  by Martin Eisemann, Thorsten Grosch, Stefan Müller and Marcus Magnor
  Computer Graphics Lab, TU Braunschweig, Germany and
  University of Koblenz-Landau, Germany

  Parts of this code are taken from
  "Fast Ray-Axis Aligned Bounding Box Overlap Tests With Pluecker Coordinates"
  by Jeffrey Mahovsky and Brian Wyvill
  Department of Computer Science, University of Calgary

  This source code is public domain, but please mention us if you use it.

******************************************************************************/
#include <math.h>
#include "ray_intersection.h"

void make_ray(double *origin, double *dir, slope_ray_type *r)
{
	//common variables
	r->x = origin[0];
	r->y = origin[1];
	r->z = origin[2];
	double i = r->i = dir[0];
	double j = r->j = dir[1];
	double k = r->k = dir[2];

	double ii = 1.0/r->i;
	double ij = 1.0/r->j;
	double ik = 1.0/r->k;

	//ray slope
	r->ibyj = r->i * ij;
	r->jbyi = r->j * ii;
	r->jbyk = r->j * ik;
	r->kbyj = r->k * ij;
	r->ibyk = r->i * ik;
	r->kbyi = r->k * ii;
	r->c_xy = r->y - r->jbyi * r->x;
	r->c_xz = r->z - r->kbyi * r->x;
	r->c_yx = r->x - r->ibyj * r->y;
	r->c_yz = r->z - r->kbyj * r->y;
	r->c_zx = r->x - r->ibyk * r->z;
	r->c_zy = r->y - r->jbyk * r->z;

	//ray slope classification
	if(i < 0)
	{
		if(j < 0)
		{
			if(k < 0)
			{
				r->classification = MMM;
			}
			else if(k > 0){
				r->classification = MMP;
			}
			else//(k >= 0)
			{
				r->classification = MMO;
			}
		}
		else//(j >= 0)
		{
			if(k < 0)
			{
				r->classification = MPM;
				if(j==0)
					r->classification = MOM;
			}
			else//(k >= 0)
			{
				if((j==0) && (k==0))
					r->classification = MOO;
				else if(k==0)
					r->classification = MPO;
				else if(j==0)
					r->classification = MOP;
				else
					r->classification = MPP;
			}
		}
	}
	else//(i >= 0)
	{
		if(j < 0)
		{
			if(k < 0)
			{
				r->classification = PMM;
				if(i==0)
					r->classification = OMM;
			}
			else//(k >= 0)
			{
				if((i==0) && (k==0))
					r->classification = OMO;
				else if(k==0)
					r->classification = PMO;
				else if(i==0)
					r->classification = OMP;
				else
					r->classification = PMP;
			}
		}
		else//(j >= 0)
		{
			if(k < 0)
			{
				if((i==0) && (j==0))
					r->classification = OOM;
				else if(i==0)
					r->classification = OPM;
				else if(j==0)
					r->classification = POM;
				else
					r->classification = PPM;
			}
			else//(k > 0)
			{
				if(i==0)
				{
					if(j==0)
						r->classification = OOP;
					else if(k==0)
						r->classification = OPO;
					else
						r->classification = OPP;
				}
				else
				{
					if((j==0) && (k==0))
						r->classification = POO;
					else if(j==0)
						r->classification = POP;
					else if(k==0)
						r->classification = PPO;
					else
						r->classification = PPP;
				}
			}
		}
	}
}

extern "C" void make_slope_ray_(cpp_src_ray_type *r_in, slope_ray_type *r) {
	make_ray(r_in->start, r_in->dir, r);
	r->length = r_in->length;
}
