#pragma once

#define MORTON_THREADS_PER_BLOCK 512
#define BUILD_THREADS_PER_BLOCK 512
#define AABB_THREADS_PER_BLOCK 512
#define TRACE_THREADS_PER_BLOCK 256
#define MAX_BLOCKS 112 // 7MPs * 16 blocks/MP for compute capability 3.0.
