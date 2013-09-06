namespace grace {

template <typename Integer, typename Float>
class BinaryRadixTree
{
    const Float *primitives_;
    const int nPrimitives_;
    const int nNodes_;
    const int nLeaves_;
    thrust::device_vector<Node> nodes_;
    thrust::device_vector<Leaf> leaves_;

public:
    BinaryRadixTree(const Float *primitives, const int nPrimitives);
    ~BinaryRadixTree();
    void build(void);

};

} // namespace grace
