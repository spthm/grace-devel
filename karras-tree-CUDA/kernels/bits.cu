#include "../types.h"
#include "bits.cuh"

namespace grace {

namespace gpu {

template UInteger32 bit_prefix<UInteger32>(UInteger32 a, UInteger32 b);
template UInteger64 bit_prefix<UInteger64>(UInteger64 a, UInteger64 b);
}

} // namespace gpu

} // namespace grace
