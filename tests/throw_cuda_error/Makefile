include ../Makefile.common

all: throw_cuda_error

%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

throw_cuda_error: throw_cuda_error.o
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f throw_cuda_error

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
