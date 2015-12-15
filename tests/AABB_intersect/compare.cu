#include "compare.cuh"

#include <iostream>
#include <stdexcept>

int compare_hitcounts(const thrust::host_vector<int>& hits_a,
                      const std::string name_a,
                      const thrust::host_vector<int>& hits_b,
                      const std::string name_b,
                      const bool verbose)
{
    if (hits_a.size() != hits_b.size()) {
        throw std::invalid_argument("Hit vectors are different sizes");
    }

    size_t errors = 0;
    for (size_t i = 0; i < hits_a.size(); ++i)
    {
        if (hits_a[i] != hits_b[i])
        {
            ++errors;

            if (!verbose) {
                continue;
            }

            std::cout << "Ray " << i << ": " << name_a << " != " << name_b
                      << "  (" << hits_a[i] << " != " << hits_b[i] << ")"
                      << std::endl;
        }
    }
    if (errors != 0) {
        std::cout << name_a << " != " << name_b << " (" << errors << " case"
                  << (errors > 1 ? "s)" : ")") << std::endl;
    }
    else {
        std::cout << name_a << " == " << name_b << " for all ray-AABB pairs"
                  << std::endl;
    }
    if (verbose) {
        std::cout << std::endl;
    }

    return errors;
}
