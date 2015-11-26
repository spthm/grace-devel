include ../Makefile.common

all: AABB_intersect

AABB_intersect: AABB_intersect.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f AABB_intersect

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
