include ../Makefile.common

all: profile_one_to_many_rays_gadget

profile_one_to_many_rays_gadget: profile_one_to_many_rays_gadget.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f profile_one_to_many_rays_gadget

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
