include ../Makefile.common

_DEPS = print_table.h
DEPS += $(_DEPS)
NVCCDEPS += $(_DEPS)

OBJS = print_table.o benchmark_rngs.o

all: benchmark_rngs

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

benchmark_rngs: $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f benchmark_rngs

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
