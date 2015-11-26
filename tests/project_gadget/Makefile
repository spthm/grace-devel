include ../Makefile.common

all: project_gadget

project_gadget: project_gadget.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f project_gadget

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
