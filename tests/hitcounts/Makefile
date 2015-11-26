include ../Makefile.common

all: hitcounts

hitcounts: hitcounts.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f hitcounts

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
