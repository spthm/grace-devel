include ../Makefile.common

OBJS = statistics.o
NVCCDEPS += stats_math.cuh statistics.cuh

all: ray_dump ripleyk_stats uniformity_stats

ray_dump: ray_dump.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

ripleyk_stats: ripleyk_stats.cu $(NVCCDEPS) $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(OBJS) $(NVCCLFLAGS) $(NVCCLIBS)

uniformity_stats: uniformity_stats.cu $(NVCCDEPS) $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(OBJS) $(NVCCLFLAGS) $(NVCCLIBS)

%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f ray_dump
	@-rm -f ripleyk_stats
	@-rm -f uniformity_stats

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
