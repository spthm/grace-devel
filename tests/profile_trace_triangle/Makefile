include ../Makefile.common

OBJS = ../helper/vector_math.o profile_trace_triangle.o
NVCCDEPS += triangle.cuh ../helper/vector_math.cuh


all: profile_trace_triangle

# Note we need -dc in order to link device code from multiple .cu files.
%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -dc $< -o $@

profile_trace_triangle: $(OBJS) $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -o $@ $(OBJS) $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f profile_trace_triangle

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
