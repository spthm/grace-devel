include ../Makefile.common

_CXXFLAGS = -g
_INCLUDES = -I./
_DEPS = vector4_host.h vector4_device.h vector4_common.h Makefile

CXXFLAGS += $(_CXXFLAGS)
INCLUDES += $(_INCLUDES)
DEPS += $(_DEPS)

NVCCINCLUDES += $(_INCLUDES)
NVCCDEPS += $(_DEPS)


all: vector4
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

vector4: vector4.o vector4_host.o vector4_device.o
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f vector4

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
