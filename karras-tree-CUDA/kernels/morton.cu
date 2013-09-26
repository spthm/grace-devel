#include "../types.h"
#include "bits.cuh"
#include "morton.cuh"

namespace grace {

namespace gpu {

// Explicitly instantiate the morton_key_functor templates for these
// parameter types only.
template class morton_key_functor<UInteger32, float>;
template class morton_key_functor<UInteger32, double>;
template class morton_key_functor<UInteger32, long double>;
template class morton_key_functor<UInteger64, float>;
template class morton_key_functor<UInteger64, double>;
template class morton_key_functor<UInteger64, long double>;

} // namespace gpu

} // namespace grace
