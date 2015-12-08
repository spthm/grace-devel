#include <thrust/host_vector.h>

#include <string>

int compare_hitcounts(const thrust::host_vector<int>& hits_a,
                      const std::string name_a,
                      const thrust::host_vector<int>& hits_b,
                      const std::string name_b,
                      const bool verbose = true);
