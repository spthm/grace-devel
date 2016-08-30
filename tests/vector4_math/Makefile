include ../Makefile.common

_CXXFLAGS = -g

CXXFLAGS += $(_CXXFLAGS)

all: vector4_math
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

vector4_math: vector4_math.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f vector4_math

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
