include ../Makefile.common

all: aligned_allocator

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

aligned_allocator: aligned_allocator.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(CXXLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f aligned_allocator

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
