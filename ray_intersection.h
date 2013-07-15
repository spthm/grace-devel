typedef struct {
  double start[3];
  double dir[3];
  double length;
  int dir_class;
} cpp_src_ray_type;

typedef struct {
  //common variables
  int classification; // This must be near the top or weird shit happens...
  double x, y, z;   // ray origin
  double i, j, k;   // ray direction

  // ray slope
  double ibyj, jbyi, kbyj, jbyk, ibyk, kbyi; //slope
  double c_xy, c_xz, c_yx, c_yz, c_zx, c_zy;
  double length;

  // SPHRAY src_ray_type variables
  double freq, enrg, pcnt, pini, dt_s;
} slope_ray_type;

/* The 'O' implies that this component is zero, and essentially never happens
 * when using randomly generated ray directions, so only classifications [0,7]
 * occur - as for the Pluecker case.  Might be able to make use of this later to
 * simplify the make_ray function.
*/
enum CLASSIFICATION
{ MMM, MMP, MPM, MPP, PMM, PMP, PPM, PPP, POO, MOO, OPO, OMO, OOP, OOM,
  OMM,OMP,OPM,OPP,MOM,MOP,POM,POP,MMO,MPO,PMO,PPO };
