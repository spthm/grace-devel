include ../Makefile.common

all: bvh_consistency

bvh_consistency: bvh_consistency.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f bvh_consistency

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
