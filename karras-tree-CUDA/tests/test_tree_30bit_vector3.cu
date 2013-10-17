#include <iostream>
#include <iomanip>
#include <bitset>
#include <cstdlib>

#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include "../types.h"
#include "../nodes.h"
#include "../kernels/bintree_build_kernels_vector3.cuh"

// See:
// https://code.google.com/p/thrust/source/browse/examples/monte_carlo.cu
// as well as:
// http://www.reedbeta.com/blog/2013/01/12/quick-and-easy-gpu-random-numbers-in-d3d11/
// and links therin.


// Thomas Wang hash.
__host__ __device__ unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

class random_vector3_functor
{
    // Constructed on the host.
    grace::Vector3<float> random_vector3;

public:
    __host__ __device__ grace::Vector3<float> operator() (unsigned int n)
    {
        unsigned int seed = hash(n);
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<float> u01(0,1);


        random_vector3.x = u01(rng);
        random_vector3.y = u01(rng);
        random_vector3.z = u01(rng);

        return random_vector3;
    }
};

class random_float_functor
{
public:
    __host__ __device__ float operator() (unsigned int n)
    {

        unsigned int seed = hash(n);
        thrust::default_random_engine rng(seed);
        thrust::uniform_real_distribution<float> u0p1(0,0.1);

        return u0p1(rng);
    }
};

int main(int argc, char* argv[]) {

    typedef grace::Vector3<float> Vector3f;

    /* Generate N random position vectors, i.e. 3*N random floats in [0,1) */

    unsigned int N = (unsigned int) std::strtol(argv[1], NULL, 10);
    std::cout << "Will generate " << N << " random points." << std::endl;
    thrust::device_vector<Vector3f> d_centres(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_centres.begin(),
                      random_vector3_functor() );


    /* Generate N random radii as floats in [0,1). */

    thrust::device_vector<float> d_radii(N);

    thrust::transform(thrust::counting_iterator<unsigned int>(0),
                      thrust::counting_iterator<unsigned int>(N),
                      d_radii.begin(),
                      random_float_functor() );


    /* Generate the Morton key of each position. */

    thrust::device_vector<UInteger32> d_keys(N);
    Vector3f bottom(0., 0., 0.);
    Vector3f top(1., 1., 1.);

    thrust::transform(d_centres.begin(),
                      d_centres.end(),
                      d_keys.begin(),
                      grace::morton_key_functor<UInteger32, float>(bottom, top) );


    /* Sort the position vectors by their keys.
     */

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_centres.begin());


    thrust::host_vector<grace::Vector3<float> > h_centres = d_centres;
    thrust::host_vector<float> h_radii = d_radii;
    thrust::host_vector<UInteger32> h_keys = d_keys;
    for (int i=0; i<N; i++) {
        std::cout << "x: " << std::fixed << std::setw(15) << std::setprecision(15)
                           << std::setfill('0') << h_centres[i].x << std::endl;
        std::cout << "y: " << std::fixed << std::setw(15) << std::setprecision(15)
                           << std::setfill('0') << h_centres[i].y << std::endl;
        std::cout << "z: " << std::fixed << std::setw(15) << std::setprecision(15)
                           << std::setfill('0') << h_centres[i].z << std::endl;
        std::cout << "r: " << std::fixed << std::setw(15) << std::setprecision(15)
                           << std::setfill('0') << h_radii[i] << std::endl;
        std::cout << "Key: " << (std::bitset<32>) h_keys[i] << std::endl;
        std::cout << std::endl;
    }

    /* Build the tree from the keys. */

    thrust::device_vector<grace::Node> d_nodes(N-1);
    thrust::device_vector<grace::Leaf> d_leaves(N);

    grace::build_nodes(d_nodes, d_leaves, d_keys);
    grace::find_AABBs(d_nodes, d_leaves, d_centres, d_radii);

    thrust::host_vector<grace::Node> h_nodes = d_nodes;
    thrust::host_vector<grace::Leaf> h_leaves = d_leaves;
    std::cout << "Nodes:\n" << std::endl;
    for (int i=0; i<(N-1); i++) {
        std::cout << "i:               " << i << std::endl;
        std::cout << "left leaf flag:  "
                  << (h_nodes[i].left_leaf_flag ? "True" : "False") << std::endl;
        std::cout << "left:   " << h_nodes[i].left << std::endl;
        std::cout << "right leaf flag: "
                  << (h_nodes[i].right_leaf_flag ? "True" : "False") << std::endl;
        std::cout << "right:           " << h_nodes[i].right << std::endl;
        std::cout << "parent:          " << h_nodes[i].parent << std::endl;
        std::cout << std::endl;
    }
    std::cout << "Leaves:\n" << std::endl;
    for (int i=0; i<N; i++) {
        std::cout << "i:      " << i << std::endl;
        std::cout << "parent: " << h_leaves[i].parent << std::endl;
        std::cout << std::endl;
    }


}
