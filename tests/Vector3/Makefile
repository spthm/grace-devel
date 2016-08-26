include ../Makefile.common

_CXXFLAGS = -g
_INCLUDES = -I./
_DEPS = vector3_host.h vector3_device.h vector3_common.h ../helper-unit/test_types.h Makefile

CXXFLAGS += $(_CXXFLAGS)
INCLUDES += $(_INCLUDES)
DEPS += $(_DEPS)

NVCCINCLUDES += $(_INCLUDES)
NVCCDEPS += $(_DEPS)


all: vector3
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

vector3: vector3.o vector3_host.o vector3_device.o
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f vector3

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
