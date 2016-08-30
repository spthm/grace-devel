include ../Makefile.common

_CXXFLAGS = -g

CXXFLAGS += $(_CXXFLAGS)

all: AABB
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

AABB: AABB.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f AABB

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
