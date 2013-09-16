namespace grace {

template <typename UInteger>
void build_nodes(thrust::device_vector<Node> d_nodes,
                 thrust::device_vector<Leaf> d_leaves,
                 thrust::device_vector<UInteger> d_keys);

template <typename UInteger>
void find_AABBs(thrust::device_vector<Node> d_nodes,
                thrust::device_vector<Leaf> d_leaves,
                thrust::device_vector<float> d_sphere_centres,
                thrust::device_vector<float> d_sphere_radii);


} // namespace grace
