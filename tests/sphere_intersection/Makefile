include ../Makefile.common

# Ensures float/double calculations obey IEEE binary32/64 rules, rather than
# being computed by the x87 FPU.
_FLAGS = -fopenmp -mfpmath=sse -msse2
_INCLUDES = -I./ -I$(GMP_INC_DIR)
_LFLAGS = -L$(GMP_LIB_DIR)
_LIBS = -lgmp -lgmpxx
_DEPS = interval.h quadratic.h intersection.cuh Makefile

CXXFLAGS += $(_FLAGS)
INCLUDES += $(_INCLUDES)
LFLAGS += $(_LFLAGS)
LIBS += $(_DEPS)
DEPS += $(_DEPS)

NVCCFLAGS += -Xcompiler "$(_FLAGS)"
NVCCINCLUDES += $(_INCLUDES)
NVCCLFLAGS += $(_LFLAGS)
NVCCLIBS += $(_LIBS)
NVCCDEPS += $(_DEPS)


all: sphere_intersection

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

sphere_intersection: sphere_intersection.o interval.o intersection.o
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f sphere_intersection

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
