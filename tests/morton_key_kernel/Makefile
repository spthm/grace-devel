include ../Makefile.common

all: 30bit_keys 63bit_keys

30bit_keys: 30bit_keys.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

63bit_keys: 63bit_keys.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f 30bit_keys
	@-rm -f 63bit_keys

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
