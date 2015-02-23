#pragma once

namespace grace {

const int MORTON_THREADS_PER_BLOCK = 512;
const int BUILD_THREADS_PER_BLOCK = 512;
const int SHIFTS_THREADS_PER_BLOCK = 512;
const int AABB_THREADS_PER_BLOCK = 512;
const int TRACE_THREADS_PER_BLOCK = 256;
const int RAYS_THREADS_PER_BLOCK = 512;
const int MAX_BLOCKS = 112; // 7MPs * 16 blocks/MP for compute capability 3.0.
const int WARP_SIZE = 32;
const int STACK_SIZE = 64;

} // namespace grace
