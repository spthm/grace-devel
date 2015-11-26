include ../Makefile.common

all: distance_sort

distance_sort: distance_sort.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f distance_sort

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
