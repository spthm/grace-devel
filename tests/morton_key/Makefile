include ../Makefile.common

all: 30bit_key 63bit_key

30bit_key: 30bit_key.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

63bit_key: 63bit_key.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f 30bit_key
	@-rm -f 63bit_key

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
