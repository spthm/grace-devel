include ../Makefile.common

all: tree

tree: tree.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f tree

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
