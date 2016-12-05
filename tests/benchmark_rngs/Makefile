include ../Makefile.common

all: benchmark_rngs

benchmark_rngs: benchmark_rngs.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f benchmark_rngs

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
