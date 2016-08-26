include ../Makefile.common

all: aligned_malloc

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

aligned_malloc: aligned_malloc.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(CXXLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f aligned_malloc

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
