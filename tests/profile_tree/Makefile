include ../Makefile.common

all: profile_tree

profile_tree: profile_tree.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f profile_tree

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
