include ../Makefile.common

all: profile_project_gadget

profile_project_gadget: profile_project_gadget.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f profile_project_gadget

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
