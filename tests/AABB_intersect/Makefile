include ../Makefile.common

# AABB_intersect.o should be first, otherwise the linker might throw away
# symbols we need.
OBJS = AABB_intersect.o \
AABB.o \
ray.o \
compare.o \
aila_laine_karras.o \
eisemann.o \
plucker.o \
williams.o

NVCCDEPS += auxillary.cuh intersectors.cuh profile.cuh

# _FORCE_INLINES is required to succesfully compile AABB.cu when using newer
# versions of gcc (seemingly 4.9+).
NVCCFLAGS += -O3 -D_FORCE_INLINES

all: AABB_intersect

AABB_intersect: $(OBJS) $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -o $@ $(OBJS) $(NVCCLFLAGS) $(NVCCLIBS)

# Note we need -dc in order to link device code from multiple .cu files.
%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -dc $< -o $@

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f AABB_intersect

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
