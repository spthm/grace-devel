CXX = g++
CPPFLAGS =

FC = gfortran
FFLAGS = -cpp

CUC = nvcc
CUFLAGS = -arch sm_30

CUDALIBS = -L/usr/local/cuda/lib64 -Wl,-rpath=/usr/local/cuda/lib64 -lcudart

EXTRALIBS = -lstdc++ -lgfortran

OBJS = mt19937.o \
	myf03.o \
	ray.o \
	ray_intersection_gpu.o \
	ray_intersection.o \
	ray_slope.o \
	ray_test.o

all: ray_test

ray_test: ${OBJS}
	${CXX} ${OBJS} ${CUDALIBS} ${EXTRALIBS} -o ray_test

%.o: %.F90
	${FC} ${FFLAGS} -c $< -o $@

%.o: %.f90
	${FC} ${FFLAGS} -c $< -o $@

%.o: %.cpp
	${CXX} ${CPPFLAGS} -c $< -o $@

%.o: %.cu
	${CUC} ${CUFLAGS} -c $< -o $@

.PHONY: clean
clean:
	rm *.o *.mod ray_test ray_test_float
