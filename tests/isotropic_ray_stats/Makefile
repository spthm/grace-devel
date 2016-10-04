include ../Makefile.common

OBJS = statistics.o ../helper/vector_math.o
NVCCDEPS += stats_math.cuh statistics.cuh ../helper/vector_math.cuh

all: ripleyk_stats uniformity_stats

ripleyk_stats: ripleyk_stats.cu $(NVCCDEPS) $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(OBJS) $(NVCCLFLAGS) $(NVCCLIBS)

uniformity_stats: uniformity_stats.cu $(NVCCDEPS) $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(OBJS) $(NVCCLFLAGS) $(NVCCLIBS)

# Note we need -dc in order to link device code from multiple .cu files.
%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -dc $< -o $@

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f ripleyk_stats
	@-rm -f uniformity_stats

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
