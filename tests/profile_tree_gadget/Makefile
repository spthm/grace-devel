include ../Makefile.common

all: profile_tree_gadget

profile_tree_gadget: profile_tree_gadget.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f profile_tree_gadget

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
