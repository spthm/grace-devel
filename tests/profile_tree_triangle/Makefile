include ../Makefile.common

OBJS = profile_tree_triangle.o \
triangle.o \
../helper/vector_math.o ../helper/read_ply.o ../helper/ply.o

NVCCDEPS += triangle.cuh \
../helper/vector_math.cuh ../helper/read_ply.hpp ../helper/ply.h

INCLUDES += -I../helper/ $(CUDA_INC)

all: profile_tree_triangle

profile_tree_triangle: $(OBJS) $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -o $@ $(OBJS) $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f profile_tree_triangle

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Note we need -dc in order to link device code from multiple .cu files.
%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -dc $< -o $@

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
