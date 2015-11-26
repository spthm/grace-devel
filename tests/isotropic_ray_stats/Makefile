include ../Makefile.common

all: ripleyk_stats uniformity_stats

ripleyk_stats: ripleyk_stats.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

uniformity_stats: uniformity_stats.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f ripleyk_stats
	@-rm -f uniformity_stats

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
