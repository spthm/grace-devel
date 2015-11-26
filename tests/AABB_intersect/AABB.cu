#include "AABB.cuh"

#include <thrust/random.h>

void random_aabbs(thrust::host_vector<AABB>& boxes,
                  float box_min, float box_max)
{
    thrust::default_random_engine rng(1234);
    thrust::uniform_real_distribution<float> uniform(box_min, box_max);

    for (size_t i = 0; i < boxes.size(); ++i)
    {
        float bx = uniform(rng);
        float by = uniform(rng);
        float bz = uniform(rng);
        float tx = uniform(rng);
        float ty = uniform(rng);
        float tz = uniform(rng);

        boxes[i].bx = min(bx, tx);
        boxes[i].by = min(by, ty);
        boxes[i].bz = min(bz, tz);

        boxes[i].tx = max(bx, tx);
        boxes[i].ty = max(by, ty);
        boxes[i].tz = max(bz, tz);
    }
}
