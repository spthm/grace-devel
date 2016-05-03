include ../Makefile.common

all: segmented_scan

sgpuutil.o: $(SGPU_DIR)/src/sgpuutil.cpp
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -o $@ -c $<

segmented_scan: segmented_scan.cu sgpuutil.o $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f segmented_scan

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
