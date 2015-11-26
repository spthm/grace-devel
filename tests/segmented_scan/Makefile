include ../Makefile.common

all: segmented_scan

segmented_scan: segmented_scan.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f segmented_scan

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
