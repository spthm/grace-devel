GRACE_FLAGS= -DGRACE_NODES_TEX -DGRACE_SPHERES_TEX

MGPU_DIR= $(HOME)/Code/moderngpu/
# MGPU_DIR= $(HOME)/Code/CUDA/moderngpu/

INCLUDES= -I $(MGPU_DIR)/include/

NVCC= nvcc

# MGPU REQUIRES that compilation targets devices you will run on.
GENCODE_20= -gencode arch=compute_20,code=sm_20
GENCODE_21= -gencode arch=compute_21,code=sm_21
GENCODE_30= -gencode arch=compute_30,code=sm_30
GENCODE_35= -gencode arch=compute_35,code=sm_35

CUDARCH= $(GENCODE_20) $(GENCODE_30)

NVCCFLAGS= $(CUDARCH) $(INCLUDES) $(GRACE_FLAGS)

NVCCLIBS= -L /opt/cuda/lib64 -lcuda -lcurand
# NVCCLIBS= -L $(HOME)/usr/local/cuda/lib64 -lcuda -lcurand


OUT_DIR= bin

DEPS= Makefile $(wildcard ../kernels/*.cuh) $(wildcard ../*.cuh) $(wildcard ../*.h)

UNIT_TEST_BIN= bit_prefix_32bit.out \
	bit_prefix_64bit.out \
	common_prefix_32bit.out \
	common_prefix_64bit.out \
	morton_30bit.out \
	morton_63bit.out \
	morton_kernel_30bit.out \
	morton_kernel_63bit.out
UNIT_TEST_BIN:= $(addprefix $(OUT_DIR)/,$(UNIT_TEST_BIN))

TEST_BIN= benchmarksegscan.out \
	sort_by_distance.out \
	trace_gadget_hproj.out \
	trace_hproj.out \
	tree_trace.out \
	uniform_rays_ripleyk.out \
	uniform_rays_stats.out
TEST_BIN:= $(addprefix $(OUT_DIR)/,$(TEST_BIN))

TEST_DUMP_BIN= project_gadget.out \
	trace_hitcounts.out \
	tree_30bit.out \
	uniform_rays_stats.out
TEST_DUMP_BIN:= $(addprefix $(OUT_DIR)/,$(TEST_DUMP_BIN))

PROFILER_BIN= AABB_speed.out \
	profile_project_gadget.out \
	profile_trace_gadget_centre.out \
	profile_tree_30bit.out \
	profile_tree_gadget_30bit.out \
	zip_sort.out
PROFILER_BIN:= $(addprefix $(OUT_DIR)/,$(PROFILER_BIN))

all: unit test dump profile

unit: $(UNIT_TEST_BIN) $(DEPS)

test: $(TEST_BIN) $(DEPS)

dump: $(TEST_DUMP_BIN) $(DEPS)

profile: $(PROFILER_BIN) $(DEPS)

mgpucontext.o: $(MGPU_DIR)/src/mgpucontext.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

mgpuutil.o: $(MGPU_DIR)/src/mgpuutil.cpp
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<

$(OUT_DIR)/%.out: %.cu $(DEPS) mgpucontext.o mgpuutil.o | $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBS) $< -o $@ mgpucontext.o mgpuutil.o

$(OUT_DIR):
	mkdir -p $(OUT_DIR)


.PHONY: clean

clean:
	-rm -r $(OUT_DIR)/*.out
	-rm *.o

distclean: clean
	-rm -r $(OUT_DIR)
