namespace grace {

namespace detail {

// If this were not a template then a class A, inheriting from multiple base
// classes B and C, which themselves inherit from NonCopyable, would have
// two subobjects of type NonCopyable. But according to the standard, two base
// subobjects of the same type are required to have different addresses within
// the object representation of the most derived type, here A. Hence empty base
// class optimization could not be applied.
// Using CRTP ensures that each base class is unique.
template <class Derived>
class NonCopyable
{
protected:
    NonCopyable () {}
    ~NonCopyable () {}
private:
    NonCopyable (const NonCopyable&);
    NonCopyable& operator=(const NonCopyable&);
};

} // namespace detail

} // namespace grace
