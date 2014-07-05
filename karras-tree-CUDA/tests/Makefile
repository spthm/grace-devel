NVCC= nvcc
CUDARCH= -arch=sm_20
NVCCFLAGS= $(CUDARCH)
NVCCLIBS= -L /home/spth/usr/local/cuda/lib64 -lcuda -lcurand
#NVCCLIBS= -L /opt/cuda/lib64 -lcuda -lcurand

OUT_DIR= bin

DEPS= Makefile $(wildcard ../kernels/*.cuh)

UNIT_TEST_BIN= bit_prefix_32bit.out \
	bit_prefix_64bit.out \
	common_prefix_32bit.out \
	common_prefix_64bit.out \
	morton_30bit.out \
	morton_63bit.out \
	morton_kernel_30bit.out \
	morton_kernel_63bit.out
UNIT_TEST_BIN:= $(addprefix $(OUT_DIR)/,$(UNIT_TEST_BIN))

TEST_BIN= sort_by_distance.out \
	trace_gadget_hproj.out \
	trace_hproj.out \
	tree_trace.out \
	uniform_rays.out
TEST_BIN:= $(addprefix $(OUT_DIR)/,$(TEST_BIN))

TEST_DUMP_BIN= project_gadget.out trace_hitcounts.out tree_30bit.out
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

directories:
	mkdir -p $(OUT_DIR)

$(OUT_DIR)/%.out: %.cu $(DEPS) | $(OUT_DIR)
	$(NVCC) $(NVCCFLAGS) $(NVCCLIBS) $< -o $@

$(OUT_DIR):
	mkdir -p $(OUT_DIR)


.PHONY: clean

clean:
	-rm -r bin/*.out

distclean: clean
	-rm -r bin
