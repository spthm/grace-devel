include ../Makefile.common

OBJS = render_triangle.o \
tris_trace.o \
tris_render.o \
triangle.o \
../helper/read_ply.o ../helper/ply.o

NVCCDEPS += triangle.cuh \
tris_trace.cuh \
tris_tree.cuh \
tris_render.cuh \
../helper/read_ply.hpp ../helper/ply.h

INCLUDES += -I../helper/ $(CUDA_INC)


all: render_triangle

%.o: %.c $(DEPS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Note we need -dc in order to link device code from multiple .cu files.
%.o: %.cu $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -dc $< -o $@

render_triangle: $(OBJS) $(NVCCDEPS)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -o $@ $(OBJS) $(NVCCLFLAGS) $(NVCCLIBS)

clean:
	@-rm -f *.o

distclean: clean
	@-rm -f render_triangle

.PHONY: clean
.PHONY: distclean

.DEFAULT_GOAL := all
