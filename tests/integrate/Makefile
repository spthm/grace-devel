include ../Makefile.common

all: integrate

integrate: integrate.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f integrate

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
