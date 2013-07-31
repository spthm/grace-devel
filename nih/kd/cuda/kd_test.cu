/*
 * Copyright (c) 2010-2011, NVIDIA Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of NVIDIA Corporation nor the
 *     names of its contributors may be used to endorse or promote products
 *     derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <nih/kd/cuda/kd_builder.h>
#include <nih/kd/cuda/kd_context.h>
#include <nih/kd/cuda/knn.h>
#include <nih/sampling/random.h>
#include <nih/time/timer.h>
#include <nih/basic/cuda_domains.h>
#include <nih/tree/model.h>
#include <nih/tree/cuda/reduce.h>

namespace nih {

namespace {

bool check_point(
    const uint32    point_idx,
    const Vector4f  point,
    const Kd_node*  nodes,
    const uint2*    leaves,
    const uint2*    ranges)
{
    uint32 node_index = 0;

    bool success = true;

    while (1)
    {
        const Kd_node node  = nodes[ node_index ];
        const uint2   range = ranges[ node_index ];

        if (point_idx < range.x || point_idx >= range.y)
        {
            success = false;
            break;
        }

        if (node.is_leaf())
        {
            const uint2 leaf = leaves[ node.get_leaf_index() ];
            success = (point_idx >= leaf.x && point_idx < leaf.y);
            break;
        }
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            node_index = node.get_child_offset() + 
                (point[ split_dim ] < split_plane ? 0u : 1u);
        }
    }

    if (success)
        return true;

    // error logging
    fprintf(stderr, "idx : %u\n", point_idx);
    fprintf(stderr, "p   : %f, %f, %f\n", point[0], point[1], point[2]);

    node_index = 0;

    while (1)
    {
        const Kd_node node  = nodes[ node_index ];
        const uint2   range = ranges[ node_index ];
        fprintf(stderr, "\n");
        fprintf(stderr, "node   : %u\n", node_index);
        fprintf(stderr, "range  : [%u, %u)\n", range.x, range.y);

        if (point_idx < range.x || point_idx >= range.y)
        {
            fprintf(stderr, "out of range!\n");
            return false;
        }

        if (node.is_leaf())
        {
            const uint2 leaf = leaves[ node.get_leaf_index() ];
            fprintf(stderr, "leaf : %u = [%u,%u)\n", node.get_leaf_index(), leaf.x, leaf.y);
            fprintf(stderr, "out of range!\n");
            return (point_idx >= leaf.x && point_idx < leaf.y);
        }
        else
        {
            const uint32 split_dim   = node.get_split_dim();
            const float  split_plane = node.get_split_plane();

            fprintf(stderr, "dim    : %u\n", split_dim);
            fprintf(stderr, "plane  : %f\n", split_plane);
            fprintf(stderr, "offset : %u\n", node.get_child_offset());

            node_index = node.get_child_offset() + 
                (point[ split_dim ] < split_plane ? 0u : 1u);
        }
    }
}

bool check_tree(
    const uint32    n_points,
    const Vector4f* points,
    const Kd_node*  nodes,
    const uint2*    leaves,
    const uint2*    ranges)
{
    for (uint32 i = 0; i < n_points; ++i)
    {
        if (check_point(
            i,
            points[i],
            nodes,
            leaves,
            ranges ) == false)
            return false;
    }
    return true;
}

bool check_knn(
    const uint32                   n_points,
    const cuda::Kd_knn<3>::Result* results)
{
    for (uint32 i = 0; i < n_points; ++i)
    {
        if (results[i].dist2 != 0.0f)
            return false;
    }
    return true;
}

template <uint32 K>
void test_knn(
    const uint32            n_test_points,
    const Kd_node*          kd_nodes,
    const uint2*            kd_ranges,
    const uint2*            kd_leaves,
    const Vector4f*         kd_points,
    cuda::Kd_knn_result*    results)
{
    fprintf(stderr, "k-d tree %u-NN test... started\n", K);

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cuda::Kd_knn<3> knn;

    float time = 0.0f;

    const uint32 n_tests = 100;

    for (uint32 i = 0; i <= n_tests; ++i)
    {
        float dtime;
        cudaEventRecord( start, 0 );

        knn.run<K>(
            kd_points,
            kd_points + n_test_points,
            kd_nodes,
            kd_ranges,
            kd_leaves,
            kd_points,
            results );

        cudaEventRecord( stop, 0 );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &dtime, start, stop );

        if (i) // skip the first run
            time += dtime;
    }
    time /= 1000.0f * float(n_tests);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    fprintf(stderr, "k-d tree %u-NN test... done\n", K);
    fprintf(stderr, "  points/sec : %f M\n", (n_test_points / time) / 1.0e6f );
    fprintf(stderr, "  time       : %f ms\n", time * 1000.0f );
}

template <>
void test_knn<1>(
    const uint32            n_test_points,
    const Kd_node*          kd_nodes,
    const uint2*            kd_ranges,
    const uint2*            kd_leaves,
    const Vector4f*         kd_points,
    cuda::Kd_knn_result*    results)
{
    fprintf(stderr, "k-d tree 1-NN test... started\n");

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cuda::Kd_knn<3> knn;

    float time = 0.0f;

    const uint32 n_tests = 100;

    for (uint32 i = 0; i <= n_tests; ++i)
    {
        float dtime;
        cudaEventRecord( start, 0 );

        knn.run(
            kd_points,
            kd_points + n_test_points,
            kd_nodes,
            kd_ranges,
            kd_leaves,
            kd_points,
            results );

        cudaEventRecord( stop, 0 );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &dtime, start, stop );

        if (i) // skip the first run
            time += dtime;
    }
    time /= 1000.0f * float(n_tests);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );
/*
    thrust::host_vector<cuda::Kd_knn_result> h_results( d_results );
    if (check_knn( n_test_points, thrust::raw_pointer_cast( &h_results.front() ) ) == false)
    {
        fprintf(stderr, "k-d tree 1-NN test... *** failed ***\n");
        exit(1);
    }
*/
    fprintf(stderr, "k-d tree 1-NN test... done\n");
    fprintf(stderr, "  points/sec : %f M\n", (n_test_points / time) / 1.0e6f );
    fprintf(stderr, "  time       : %f ms\n", time * 1000.0f );
}

} // anonymous namespace

void kd_test()
{
    fprintf(stderr, "k-d tree test... started\n");

    const uint32 n_points = 4*1024*1024;
    const uint32 n_tests = 100;

    thrust::host_vector<Vector4f> h_points( n_points );

    Random random;
    for (uint32 i = 0; i < n_points; ++i)
        h_points[i] = Vector4f( random.next(), random.next(), random.next(), 1.0f );

    thrust::device_vector<Vector4f> d_points( h_points );

    thrust::device_vector<Kd_node>  kd_nodes;
    thrust::device_vector<uint2>    kd_leaves;
    thrust::device_vector<uint2>    kd_ranges;
    thrust::device_vector<uint32>   kd_index;

    cuda::Kd_context context( &kd_nodes, &kd_leaves, &kd_ranges );
    cuda::Kd_builder<uint64> builder;

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    float time = 0.0f;

    for (uint32 i = 0; i <= n_tests; ++i)
    {
        float dtime;
        cudaEventRecord( start, 0 );

        builder.build(
            context,
            kd_index,
            Bbox3f( Vector3f(0.0f), Vector3f(1.0f) ),
            d_points.begin(),
            d_points.end(),
            8u );

        cudaEventRecord( stop, 0 );
        cudaEventSynchronize( stop );
        cudaEventElapsedTime( &dtime, start, stop );

        if (i) // skip the first run
            time += dtime;
    }
    time /= 1000.0f * float(n_tests);

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

    thrust::host_vector<uint32>   h_kd_index( kd_index );
    thrust::host_vector<Kd_node>  h_kd_nodes( kd_nodes );
    thrust::host_vector<uint2>    h_kd_leaves( kd_leaves );
    thrust::host_vector<uint2>    h_kd_ranges( kd_ranges );

    thrust::host_vector<Vector4f> h_sorted_points( n_points );

    thrust::gather(
        h_kd_index.begin(),
        h_kd_index.begin() + n_points,
        h_points.begin(),
        h_sorted_points.begin() );

    if (check_tree(
        n_points,
        thrust::raw_pointer_cast( &h_sorted_points.front() ),
        thrust::raw_pointer_cast( &h_kd_nodes.front() ),
        thrust::raw_pointer_cast( &h_kd_leaves.front() ),
        thrust::raw_pointer_cast( &h_kd_ranges.front() ) ) == false)
    {
        fprintf(stderr, "k-d tree test... *** failed ***\n");
        exit(1);
    }

    fprintf(stderr, "k-d tree test... done\n");
    fprintf(stderr, "  time       : %f ms\n", time * 1000.0f );
    fprintf(stderr, "  points/sec : %f M\n", (n_points / time) / 1.0e6f );

    fprintf(stderr, "  nodes  : %u\n", builder.m_node_count );
    fprintf(stderr, "  leaves : %u\n", builder.m_leaf_count );
    for (uint32 level = 0; level < 60; ++level)
        fprintf(stderr, "  level %u : %u nodes\n", level, builder.m_levels[level+1] - builder.m_levels[level] );

    // do a k-nn test
    d_points = h_sorted_points;

    uint32 n_test_points = 256*1024;

    thrust::device_vector<cuda::Kd_knn_result> d_results( n_test_points*64 );

    test_knn<1>(
        n_test_points,
        thrust::raw_pointer_cast( &kd_nodes.front() ),
        thrust::raw_pointer_cast( &kd_ranges.front() ),
        thrust::raw_pointer_cast( &kd_leaves.front() ),
        thrust::raw_pointer_cast( &d_points.front() ),
        thrust::raw_pointer_cast( &d_results.front() ) );

    test_knn<4>(
        n_test_points,
        thrust::raw_pointer_cast( &kd_nodes.front() ),
        thrust::raw_pointer_cast( &kd_ranges.front() ),
        thrust::raw_pointer_cast( &kd_leaves.front() ),
        thrust::raw_pointer_cast( &d_points.front() ),
        thrust::raw_pointer_cast( &d_results.front() ) );

    test_knn<8>(
        n_test_points,
        thrust::raw_pointer_cast( &kd_nodes.front() ),
        thrust::raw_pointer_cast( &kd_ranges.front() ),
        thrust::raw_pointer_cast( &kd_leaves.front() ),
        thrust::raw_pointer_cast( &d_points.front() ),
        thrust::raw_pointer_cast( &d_results.front() ) );

    test_knn<16>(
        n_test_points,
        thrust::raw_pointer_cast( &kd_nodes.front() ),
        thrust::raw_pointer_cast( &kd_ranges.front() ),
        thrust::raw_pointer_cast( &kd_leaves.front() ),
        thrust::raw_pointer_cast( &d_points.front() ),
        thrust::raw_pointer_cast( &d_results.front() ) );

    test_knn<32>(
        n_test_points,
        thrust::raw_pointer_cast( &kd_nodes.front() ),
        thrust::raw_pointer_cast( &kd_ranges.front() ),
        thrust::raw_pointer_cast( &kd_leaves.front() ),
        thrust::raw_pointer_cast( &d_points.front() ),
        thrust::raw_pointer_cast( &d_results.front() ) );

    test_knn<64>(
        n_test_points,
        thrust::raw_pointer_cast( &kd_nodes.front() ),
        thrust::raw_pointer_cast( &kd_ranges.front() ),
        thrust::raw_pointer_cast( &kd_leaves.front() ),
        thrust::raw_pointer_cast( &d_points.front() ),
        thrust::raw_pointer_cast( &d_results.front() ) );

    test_knn<512>(
        n_test_points / 16,
        thrust::raw_pointer_cast( &kd_nodes.front() ),
        thrust::raw_pointer_cast( &kd_ranges.front() ),
        thrust::raw_pointer_cast( &kd_leaves.front() ),
        thrust::raw_pointer_cast( &d_points.front() ),
        thrust::raw_pointer_cast( &d_results.front() ) );
}

} // namespace nih

