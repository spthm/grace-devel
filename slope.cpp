/******************************************************************************

  This source code accompanies the Journal of Graphics Tools paper:

  "Fast Ray / Axis-Aligned Bounding Box Overlap Tests using Ray Slopes" 
  by Martin Eisemann, Thorsten Grosch, Stefan Müller and Marcus Magnor
  Computer Graphics Lab, TU Braunschweig, Germany and
  University of Koblenz-Landau, Germany
  
  This source code is public domain, but please mention us if you use it.

******************************************************************************/
#include "slope.h"

bool slope(ray *r, double *bot, double *top){

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
			return false;
		
		return true;

	case MMP:
		
		if ((r->x < bot[0]) || (r->y < bot[1]) || (r->z > top[2])
			|| (r->jbyi * bot[0] - top[1] + r->c_xy > 0)
			|| (r->ibyj * bot[1] - top[0] + r->c_yx > 0)
			|| (r->jbyk * top[2] - top[1] + r->c_zy > 0)
			|| (r->kbyj * bot[1] - bot[2] + r->c_yz < 0)
			|| (r->kbyi * bot[0] - bot[2] + r->c_xz < 0)
			|| (r->ibyk * top[2] - top[0] + r->c_zx > 0)
			)
			return false;
		
		return true;

	case MPM:
		
		if ((r->x < bot[0]) || (r->y > top[1]) || (r->z < bot[2])
			|| (r->jbyi * bot[0] - bot[1] + r->c_xy < 0) 
			|| (r->ibyj * top[1] - top[0] + r->c_yx > 0)
			|| (r->jbyk * bot[2] - bot[1] + r->c_zy < 0) 
			|| (r->kbyj * top[1] - top[2] + r->c_yz > 0)
			|| (r->kbyi * bot[0] - top[2] + r->c_xz > 0)
			|| (r->ibyk * bot[2] - top[0] + r->c_zx > 0)
			)
			return false;
		
		return true;

	case MPP:
	
		if ((r->x < bot[0]) || (r->y > top[1]) || (r->z > top[2])
			|| (r->jbyi * bot[0] - bot[1] + r->c_xy < 0) 
			|| (r->ibyj * top[1] - top[0] + r->c_yx > 0)
			|| (r->jbyk * top[2] - bot[1] + r->c_zy < 0)
			|| (r->kbyj * top[1] - bot[2] + r->c_yz < 0)
			|| (r->kbyi * bot[0] - bot[2] + r->c_xz < 0)
			|| (r->ibyk * top[2] - top[0] + r->c_zx > 0)
			)
			return false;
		
		return true;

	case PMM:

		if ((r->x > top[0]) || (r->y < bot[1]) || (r->z < bot[2])
			|| (r->jbyi * top[0] - top[1] + r->c_xy > 0)
			|| (r->ibyj * bot[1] - bot[0] + r->c_yx < 0)
			|| (r->jbyk * bot[2] - top[1] + r->c_zy > 0)
			|| (r->kbyj * bot[1] - top[2] + r->c_yz > 0)
			|| (r->kbyi * top[0] - top[2] + r->c_xz > 0)
			|| (r->ibyk * bot[2] - bot[0] + r->c_zx < 0)
			)
			return false;

		return true;

	case PMP:

		if ((r->x > top[0]) || (r->y < bot[1]) || (r->z > top[2])
			|| (r->jbyi * top[0] - top[1] + r->c_xy > 0)
			|| (r->ibyj * bot[1] - bot[0] + r->c_yx < 0)
			|| (r->jbyk * top[2] - top[1] + r->c_zy > 0)
			|| (r->kbyj * bot[1] - bot[2] + r->c_yz < 0)
			|| (r->kbyi * top[0] - bot[2] + r->c_xz < 0)
			|| (r->ibyk * top[2] - bot[0] + r->c_zx < 0)
			)
			return false;

		return true;

	case PPM:

		if ((r->x > top[0]) || (r->y > top[1]) || (r->z < bot[2])
			|| (r->jbyi * top[0] - bot[1] + r->c_xy < 0)
			|| (r->ibyj * top[1] - bot[0] + r->c_yx < 0)
			|| (r->jbyk * bot[2] - bot[1] + r->c_zy < 0) 
			|| (r->kbyj * top[1] - top[2] + r->c_yz > 0)
			|| (r->kbyi * top[0] - top[2] + r->c_xz > 0)
			|| (r->ibyk * bot[2] - bot[0] + r->c_zx < 0)
			)
			return false;
		
		return true;

	case PPP:

		if ((r->x > top[0]) || (r->y > top[1]) || (r->z > top[2])
			|| (r->jbyi * top[0] - bot[1] + r->c_xy < 0)
			|| (r->ibyj * top[1] - bot[0] + r->c_yx < 0)
			|| (r->jbyk * top[2] - bot[1] + r->c_zy < 0)
			|| (r->kbyj * top[1] - bot[2] + r->c_yz < 0)
			|| (r->kbyi * top[0] - bot[2] + r->c_xz < 0)
			|| (r->ibyk * top[2] - bot[0] + r->c_zx < 0)
			)
			return false;
		
		return true;

	case OMM:

		if((r->x < bot[0]) || (r->x > top[0])
			|| (r->y < bot[1]) || (r->z < bot[2])
			|| (r->jbyk * bot[2] - top[1] + r->c_zy > 0)
			|| (r->kbyj * bot[1] - top[2] + r->c_yz > 0)
			)
			return false;

		return true;

	case OMP:

		if((r->x < bot[0]) || (r->x > top[0])
			|| (r->y < bot[1]) || (r->z > top[2])
			|| (r->jbyk * top[2] - top[1] + r->c_zy > 0)
			|| (r->kbyj * bot[1] - bot[2] + r->c_yz < 0)
			)
			return false;

		return true;

	case OPM:

		if((r->x < bot[0]) || (r->x > top[0])
			|| (r->y > top[1]) || (r->z < bot[2])
			|| (r->jbyk * bot[2] - bot[1] + r->c_zy < 0) 
			|| (r->kbyj * top[1] - top[2] + r->c_yz > 0)
			)
			return false;

		return true;

	case OPP:

		if((r->x < bot[0]) || (r->x > top[0])
			|| (r->y > top[1]) || (r->z > top[2])
			|| (r->jbyk * top[2] - bot[1] + r->c_zy < 0)
			|| (r->kbyj * top[1] - bot[2] + r->c_yz < 0)
			)
			return false;

		return true;

	case MOM:

		if((r->y < bot[1]) || (r->y > top[1])
			|| (r->x < bot[0]) || (r->z < bot[2]) 
			|| (r->kbyi * bot[0] - top[2] + r->c_xz > 0)
			|| (r->ibyk * bot[2] - top[0] + r->c_zx > 0)
			)
			return false;

		return true;

	case MOP:

		if((r->y < bot[1]) || (r->y > top[1])
			|| (r->x < bot[0]) || (r->z > top[2]) 
			|| (r->kbyi * bot[0] - bot[2] + r->c_xz < 0)
			|| (r->ibyk * top[2] - top[0] + r->c_zx > 0)
			)
			return false;

		return true;

	case POM:

		if((r->y < bot[1]) || (r->y > top[1])
			|| (r->x > top[0]) || (r->z < bot[2])
			|| (r->kbyi * top[0] - top[2] + r->c_xz > 0)
			|| (r->ibyk * bot[2] - bot[0] + r->c_zx < 0)
			)
			return false;

		return true;

	case POP:

		if((r->y < bot[1]) || (r->y > top[1])
			|| (r->x > top[0]) || (r->z > top[2])
			|| (r->kbyi * top[0] - bot[2] + r->c_xz < 0)
			|| (r->ibyk * top[2] - bot[0] + r->c_zx < 0)
			)
			return false;

		return true;

	case MMO:

		if((r->z < bot[2]) || (r->z > top[2])
			|| (r->x < bot[0]) || (r->y < bot[1]) 
			|| (r->jbyi * bot[0] - top[1] + r->c_xy > 0)
			|| (r->ibyj * bot[1] - top[0] + r->c_yx > 0)
			)
			return false;

		return true;

	case MPO:

		if((r->z < bot[2]) || (r->z > top[2])
			|| (r->x < bot[0]) || (r->y > top[1]) 
			|| (r->jbyi * bot[0] - bot[1] + r->c_xy < 0) 
			|| (r->ibyj * top[1] - top[0] + r->c_yx > 0)
			)
			return false;

		return true;

	case PMO:

		if((r->z < bot[2]) || (r->z > top[2])
			|| (r->x > top[0]) || (r->y < bot[1]) 
			|| (r->jbyi * top[0] - top[1] + r->c_xy > 0)
			|| (r->ibyj * bot[1] - bot[0] + r->c_yx < 0)  
			)
			return false;

		return true;

	case PPO:

		if((r->z < bot[2]) || (r->z > top[2])
			|| (r->x > top[0]) || (r->y > top[1])
			|| (r->jbyi * top[0] - bot[1] + r->c_xy < 0)
			|| (r->ibyj * top[1] - bot[0] + r->c_yx < 0)
			)
			return false;

		return true;

	case MOO:

		if((r->x < bot[0])
			|| (r->y < bot[1]) || (r->y > top[1])
			|| (r->z < bot[2]) || (r->z > top[2])
			)
			return false;

		return true;

	case POO:

		if((r->x > top[0])
			|| (r->y < bot[1]) || (r->y > top[1])
			|| (r->z < bot[2]) || (r->z > top[2])
			)
			return false;

		return true;

	case OMO:

		if((r->y < bot[1])
			|| (r->x < bot[0]) || (r->x > top[0])
			|| (r->z < bot[2]) || (r->z > top[2])
			)
			return false;

	case OPO:

		if((r->y > top[1])
			|| (r->x < bot[0]) || (r->x > top[0])
			|| (r->z < bot[2]) || (r->z > top[2])
			)
			return false;

	case OOM:

		if((r->z < bot[2])
			|| (r->x < bot[0]) || (r->x > top[0])
			|| (r->y < bot[1]) || (r->y > top[1])
			)
			return false;

	case OOP:

		if((r->z > top[2])
			|| (r->x < bot[0]) || (r->x > top[0])
			|| (r->y < bot[1]) || (r->y > top[1])
			)
			return false;

		return true;
	
	}

	return false;
}
