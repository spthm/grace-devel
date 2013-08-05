#ifndef BVH_H
#define BVH_H

#include "types.h"
#include "aabb.h"

struct model;

#ifndef __CUDACC__

struct bvh_node
{
    s3d::r3::box bounds;
    int first_prim_idx;
    size_t prim_count;
    unsigned split_dim; // 0->x, 1->y, 2->z
    unsigned order; // spatial order

    std::shared_ptr<bvh_node> children[2];
};
std::shared_ptr<bvh_node> make_bvh(model &model, size_t *total_nodes=NULL,
                                   size_t *height=NULL);
#endif

struct linear_8bvh_node
{
    AABB aabb;

    union
    {
        struct
        {
            unsigned children_count  : 8,
                     children_offset : 24;
        };
        unsigned children_info;
    };

    union
    {
        struct
        {
            unsigned prim_count  : 8,
                     prim_offset : 24;
        };
        unsigned prim_info;
    };
};

std::vector<linear_8bvh_node> make_linear_8bvh(model &model, 
                                               size_t *height=NULL);
#endif
