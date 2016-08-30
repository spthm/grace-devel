include ../Makefile.common

_CXXFLAGS = -g

CXXFLAGS += $(_CXXFLAGS)

all: vector_math
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

vector_math: vector_math.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f vector_math

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
