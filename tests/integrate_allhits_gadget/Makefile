include ../Makefile.common

all: integrate_allhits_gadget

integrate_allhits_gadget: integrate_allhits_gadget.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f integrate_allhits_gadget

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
