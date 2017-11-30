#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "chealpix/chealpix.h"

double randZeroToOne() {
 return rand() / (RAND_MAX + 1.);
}

int main(int argc, char *argv[]) {
 extern char *optarg;
 extern int optind, opterr, optopt;

 int c;
 size_t nRays=0;

 while( (c = getopt(argc, argv, "n:") ) != -1 ) {
  switch (c) {
   case 'n':
    nRays = strtoul(optarg,NULL,10);
    break;
   case '?':
    if(optopt=='n') {
     printf("Option -%c requires a numeric argument.\n", optopt);
    } else if (isprint (optopt)) {
     printf ("Unknown option `-%c'.\n", optopt);
    } else {
     printf ("Unknown option character `\\x%x'.\n",optopt);
     return EXIT_FAILURE;
    }
   default:
     return EXIT_FAILURE;
  }
 }

 if(nRays==0) {
  printf("generateRays -n <number of rays>\n");
 }

 /* Allocate memory for the vectors [3,N] */
 double *rayVectors = (double *)malloc(3*nRays*sizeof(double));

 long nside = npix2nside((long)nRays);
 if(nside<0) {
  printf("generateRays: ERROR: nRays must be an integer that satisfies 12 N^2\n where N is an integer\n");
  return EXIT_FAILURE;
 }
 printf("nside = %li\n",nside);

 /* Generate the ray vectors */
 for(long ipring=0; ipring<nRays; ++ipring) {
  pix2vec_nest(nside, ipring, &(rayVectors[3*ipring]));
 }
 
 for(long ipring=0; ipring<nRays; ++ipring) {
  printf("Vector[%2li]=(%6.3f,%6.3f,%6.3f)\n",ipring,
         rayVectors[3*ipring+0],
	 rayVectors[3*ipring+1],
	 rayVectors[3*ipring+2]);
 }

 /* Rotate randomly */
 /* Three angles to rotate around */
 double Tx,Ty,Tz;
 Tx = randZeroToOne()*2*M_PI;
 Ty = randZeroToOne()*2*M_PI;
 Tz = randZeroToOne()*2*M_PI;
 
 /* Rotation Matrix */
 double Rot3[9];
 Rot3[0] =  cos(Ty)*cos(Tz);
 Rot3[1] = -(cos(Ty)*sin(Tz));
 Rot3[2] =  sin(Ty);
 Rot3[3] =  cos(Tz)*sin(Tx)*sin(Ty) + cos(Tx)*sin(Tz);
 Rot3[4] =  cos(Tx)*cos(Tz) - sin(Tx)*sin(Ty)*sin(Tz);
 Rot3[5] = -(cos(Ty)*sin(Tx));
 Rot3[6] = -(cos(Tx)*cos(Tz)*sin(Ty)) + sin(Tx)*sin(Tz);
 Rot3[7] =  cos(Tz)*sin(Tx) + cos(Tx)*sin(Ty)*sin(Tz);
 Rot3[8] =  cos(Tx)*cos(Ty);

 /* Rotate the rays */
 for(long ipring=0; ipring<nRays; ++ipring) {
  double NewVec[3];
  NewVec[0] =  rayVectors[3*ipring+0]*Rot3[0]
             + rayVectors[3*ipring+1]*Rot3[1]
             + rayVectors[3*ipring+2]*Rot3[2];
  NewVec[1] =  rayVectors[3*ipring+0]*Rot3[3]
             + rayVectors[3*ipring+1]*Rot3[4]
             + rayVectors[3*ipring+2]*Rot3[5];
  NewVec[2] =  rayVectors[3*ipring+0]*Rot3[6]
             + rayVectors[3*ipring+1]*Rot3[7]
             + rayVectors[3*ipring+2]*Rot3[8];
  for(long i=0; i<3; ++i) {
   rayVectors[3*ipring+i] = NewVec[i];
  }
 }

 printf("\nAfter rotation\n");
 for(long ipring=0; ipring<nRays; ++ipring) {
  printf("Vector[%2li]=(%6.3f,%6.3f,%6.3f)\n",ipring,
         rayVectors[3*ipring+0],
	 rayVectors[3*ipring+1],
	 rayVectors[3*ipring+2]);
 }

 return EXIT_SUCCESS;
}
