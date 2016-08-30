include ../Makefile.common

_CXXFLAGS = -g

CXXFLAGS += $(_CXXFLAGS)

all: vector3_math
%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

vector3_math: vector3_math.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f vector3_math

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
