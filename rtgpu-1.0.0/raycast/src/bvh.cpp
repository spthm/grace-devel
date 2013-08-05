#include "pch.h"
#include "model.h"
#include "bvh.h"

using namespace s3d;

struct prim_info
{
    int idx;
    r3::box bounds;
    r3::point centroid;
};

const int max_prims_in_node = 10;

auto make_bvh(std::vector<prim_info> &build_data, int start, int end,
              size_t *total_nodes, std::vector<int> &ordered_prims,
              size_t curheight,
              size_t *height)
    -> std::shared_ptr<bvh_node> 
{
    assert(start >= 0 && start < build_data.size());
    assert(end > 0 && end <= build_data.size());
    assert(start < end);

    if(height)
        *height = curheight;

    if(total_nodes)
        ++*total_nodes;

    std::shared_ptr<bvh_node> node(new bvh_node);

    r3::box bounds = r3::null_box;
    for(int i=start; i<end; ++i)
        bounds |= build_data[i].bounds;

    int prim_count = end-start;

    if(prim_count == 1)
    {
        // create leaf
        node->first_prim_idx = ordered_prims.size();
        node->prim_count = prim_count;
        node->bounds = bounds;

        for(int i=start; i<end; ++i)
            ordered_prims.push_back(build_data[i].idx);
        return node;
    }

    // compute bounds of prim centroids
    r3::box centroid_bounds = r3::null_box;
    for(int i=start; i<end; ++i)
        centroid_bounds |= build_data[i].centroid;

    // choose split dimension
    unsigned split_dim = maximum_extent(centroid_bounds);

    // empty bounds?
    if(centroid_bounds.size[split_dim] == 0)
    {
        // create leaf
        node->first_prim_idx = ordered_prims.size();
        node->prim_count = prim_count;
        node->bounds = bounds;

        for(int i=start; i<end; ++i)
            ordered_prims.push_back(build_data[i].idx);

        return node;
    }

    const int bucket_count = 12;
    struct bucket_info
    {
        bucket_info() : count(0), bounds(r3::null_box) {}
        size_t count;
        r3::box bounds;
    };

    bucket_info buckets[bucket_count];

    for(int i=start; i<end; ++i)
    {
        int b = bucket_count *
            ((build_data[i].centroid[split_dim] - 
              centroid_bounds.origin[split_dim]) / 
                                    centroid_bounds.size[split_dim]);

        if(b == bucket_count)
            b = bucket_count-1;

        assert(b >= 0 && b < bucket_count);
        buckets[b].count++;
        buckets[b].bounds |= build_data[i].bounds;
    }

    // compute costs for splitting after each bucket;
    
    float cost[bucket_count-1];
    for(int i=0; i<bucket_count-1; ++i)
    {
        r3::box b0 = r3::null_box, b1 = r3::null_box;
        int count0=0, count1=1;
        for(int j=0; j<=i; ++j)
        {
            b0 |= buckets[j].bounds;
            count0 += buckets[j].count;
        }

        for(int j=i+1; j<bucket_count; ++j)
        {
            b1 |= buckets[j].bounds;
            count1 += buckets[j].count;
        }

        cost[i] = .125f + (count0*surface_area(b0) +
                           count1*surface_area(b1)) / surface_area(bounds);
    }

    // find bucket to split at that minimizes SAH metric

    float min_cost = cost[0];
    int min_cost_split=0;
    for(int i=1; i<bucket_count-1; ++i)
    {
        if(cost[i] < min_cost)
        {
            min_cost = cost[i];
            min_cost_split = i;
        }
    }

    if(prim_count < max_prims_in_node && min_cost >= prim_count)
    {
        // create leaf
        node->first_prim_idx = ordered_prims.size();
        node->prim_count = prim_count;
        node->bounds = bounds;

        for(int i=start; i<end; ++i)
            ordered_prims.push_back(build_data[i].idx);
        return node;
    }

    auto pmid = std::partition(build_data.begin()+start, build_data.begin()+end,
            [&](const prim_info &prim)
            {
                int b = bucket_count*((prim.centroid[split_dim] - 
                                      centroid_bounds.origin[split_dim]) 
                            / centroid_bounds.size[split_dim]);

                if(b == bucket_count)
                    b = bucket_count-1;
                assert(b >= 0 && b < bucket_count);
                return b <= min_cost_split;
            });

    int mid = distance(build_data.begin(), pmid);

    assert(mid >= start && mid < end);

    node->split_dim = split_dim;
    node->prim_count = 0;
    node->first_prim_idx = 0;
    node->children[0] = make_bvh(build_data, start, mid, total_nodes, 
                                 ordered_prims, curheight+1, height);
    node->children[1] = make_bvh(build_data, mid, end, total_nodes, 
                                 ordered_prims, curheight+1, height);
    node->bounds = node->children[0]->bounds | node->children[1]->bounds;

    return node;
}

auto make_bvh(model &model, size_t *total_nodes, size_t *height)
    -> std::shared_ptr<bvh_node> 
{
    if(total_nodes)
        *total_nodes = 0;

    if(height)
        *height = 0;

    if(model.faces.empty())
        return std::shared_ptr<bvh_node>();

    std::vector<prim_info> build_data;
    build_data.resize(model.faces.size());

    for(int i=0; i<model.faces.size(); ++i)
    {
        auto triangle = make_face_view(model.faces[i], model.positions);

        build_data[i].idx = i;
        auto bbox = bounds(triangle);

        build_data[i].bounds = bbox;
        build_data[i].centroid = centroid(build_data[i].bounds);
    }

    std::vector<int> ordered_prims;
    ordered_prims.reserve(build_data.size());

    auto root = make_bvh(build_data, 0, build_data.size(), total_nodes,
                         ordered_prims, 1, height);

    std::vector<math::face<int,3>> ordered_faces;
    ordered_faces.reserve(ordered_prims.size());

    for(int i=0; i<ordered_prims.size(); ++i)
    {
        assert(ordered_prims[i] >= 0 && ordered_prims[i] < model.faces.size());
        ordered_faces.push_back(model.faces[ordered_prims[i]]);
    }

    std::swap(model.faces, ordered_faces);

    return root;
}

void make_nary_level(std::vector<std::shared_ptr<bvh_node>> &linear_bvh,
                     std::shared_ptr<bvh_node> &node, 
                     int curlevel, int max_level, unsigned cur_order)
{
    // atingimos o nível ou eh um nó-folha?
    if(curlevel == max_level || node->prim_count>0)
    {
        node->order = cur_order;
        linear_bvh.push_back(node);
        return;
    }

    assert(node->children[0]);
    make_nary_level(linear_bvh, node->children[0], curlevel+1, 
                    max_level, cur_order);

    assert(node->children[1]);
    make_nary_level(linear_bvh, node->children[1], curlevel+1, 
                    max_level, cur_order | (1<<node->split_dim));
}

auto make_linear_8bvh(model &model, size_t *height) 
    -> std::vector<linear_8bvh_node>
{
    size_t node_count;
    size_t binary_height;
    auto root = make_bvh(model, &node_count, &binary_height);

    if(height)
        *height = floor(binary_height*2.0/3);

    std::vector<std::shared_ptr<bvh_node>> linear_bvh;
    linear_bvh.reserve(node_count);
    linear_bvh.push_back(root);

    std::vector<linear_8bvh_node> output_bvh;
    output_bvh.reserve(node_count);

    int arity = 8,
        log2arity = 3;

    for(int i=0; i<linear_bvh.size(); ++i)
    {
        auto node = linear_bvh[i];

        size_t child_begin = linear_bvh.size();

        // é um nó interno?
        if(node->prim_count == 0)
            make_nary_level(linear_bvh, node, 0, 3, 0);

        size_t children_count = linear_bvh.size() - child_begin;

        assert(children_count <= arity);

        auto c = centroid(node->bounds);
        auto h = node->bounds.size/2;

        assert(node->bounds.is_positive());

        linear_8bvh_node output_node;

        output_node.aabb.center.x = c.x;
        output_node.aabb.center.y = c.y;
        output_node.aabb.center.z = c.z;

        output_node.aabb.hsize.x = h.w;
        output_node.aabb.hsize.y = h.h;
        output_node.aabb.hsize.z = h.d;

        assert(!node->bounds.is_zero());

        // leaf nodes are child of themselves
        if(children_count == 0)
        {
            output_node.children_offset = i;
            output_node.children_count = 1;
        }
        else
        {
            output_node.children_offset = child_begin;
            output_node.children_count = children_count;

            sort(linear_bvh.begin()+child_begin, linear_bvh.end(),
                [](const std::shared_ptr<bvh_node> &a, 
                   const std::shared_ptr<bvh_node> &b)
                {
                    return a->order < b->order;
                });

        }

        output_node.prim_offset = node->first_prim_idx;
        output_node.prim_count = node->prim_count;

#if 0

        std::cout << "Node " << i << std::endl;
        std::cout << "\tChild count: " << children_count << std::endl;
        std::cout << "\tChild offset: " << output_node.children_offset << std::endl;
        std::cout << "\tPrim count: " << output_node.prim_count << std::endl;
        printf("\tBounds: (%f,%f,%f) - (%f,%f,%f)\n",
               output_node.aabb.center.x-output_node.aabb.hsize.x, 
               output_node.aabb.center.y-output_node.aabb.hsize.y,
               output_node.aabb.center.z-output_node.aabb.hsize.z,
               output_node.aabb.center.x+output_node.aabb.hsize.x, 
               output_node.aabb.center.y+output_node.aabb.hsize.y,
               output_node.aabb.center.z+output_node.aabb.hsize.z);
#endif

        output_bvh.push_back(output_node);

#if 0
        std::sort(linear_bvh.begin()+child_begin, linear_bvh.end(),
                  [](const std::shared_ptr<bvh_node> &a, 
                     const std::shared_ptr<bvh_node> &b) 
                      -> bool
                  {
                    return a->split_dim < b->split_dim;
                  });

        std::cout << "-------------------------------------\n";
        for(auto it = linear_bvh.begin()+child_begin; it!=linear_bvh.end(); ++it)
        {
            std::cout << centroid((*it)->bounds) << std::endl;
        }
#endif
    }

    return std::move(output_bvh);
}
