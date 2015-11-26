include ../Makefile.common

all: profile_trace_gadget

profile_trace_gadget: profile_trace_gadget.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f profile_trace_gadget

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
