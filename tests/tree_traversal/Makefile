include ../Makefile.common

# Ensures float/double calculations on the CPU obey IEEE binary32/64 rules,
# rather than being computed by the x87 FPU.
NVCCFLAGS += -Xcompiler "-fopenmp -mfpmath=sse -msse2"
# Similarly, do not let nvcc compile FMA instructions for the GPU, otherwise
# its intersection results may differ from the CPU's.
NVCCFLAGS += -fmad=false

all: tree_traversal

tree_traversal: tree_traversal.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f tree_traversal

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
