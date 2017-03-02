#pragma once

#include "grace/types.h"

namespace grace {

//
// Forward declarations
//

// Required for below.
template <typename T>
class BoundedPtr;
// Required for the friend declaration in BoundedPtr.
template <typename T>
GRACE_HOST_DEVICE typename BoundedPtr<T>::difference_type operator-(const BoundedPtr<T>&, const BoundedPtr<T>&);
template <typename T>
GRACE_HOST_DEVICE bool operator==(const BoundedPtr<T>&, const BoundedPtr<T>&);
template <typename T>
GRACE_HOST_DEVICE void swap(BoundedPtr<T>&, BoundedPtr<T>&);

// A pointer which may only access the underlying array in a specific range.
// For efficiency, range checking only occurs in debug mode.
// No checks are made on the alignment of the pointer; in order to ensure
// correct alignment, multiple BoundedPtrs should be declared in order of
// decreasing alignment requirements, just as one would do for a raw pointer.
//
// For example, if we wish to store 5 doubles, 6 floats and 2 chars, then,
// supposing char* ptr points to memory of sufficient size and satisfies the
// alignment requirements of double,
//     const size_t size = 5*sizeof(double) + 6*sizeof(float) + 2*sizeof(char)
//     bptr_d = BoundedPtr<double>(ptr, size);
//     bptr_f = BoundedPtr<float>(bptr_d + 5);
//     bptr_c = BoundedPtr<char>(bptr_f + 6);
// ensures correct alignment for all bptr_Xs, and *(bptr_c + 1) is within the
// bounds implied by ptr and size.
//
// BoundedPtr has an align_to() method to set the alignment, but note that this
// will either not modify the underlying pointer, or it will increment it, and
// hence may result in unexpected out-of-bounds accesses.
//
// Alignment requirements of a type can be checked with the GRACE_ALIGNOF macro.
template <typename T>
class BoundedPtr
{
public:
    typedef std::random_access_iterator_tag iterator_category;
    typedef T value_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef T& reference;

    template <typename U>
    friend class BoundedPtr;

    GRACE_HOST_DEVICE BoundedPtr(char* const begin, const size_t bytes);

    // end should point to one past the last valid char.
    GRACE_HOST_DEVICE BoundedPtr(char* const begin, char* const end);

    // Note this is not the copy constructor, because a copy constructor cannot
    // be a template. We use the default copy constructor. It is explicit, so it
    // is also not a converting constructor. It is explicit because converting
    // the underlying pointer from T* to U* is a non-trivial operation, and
    // should not be done implicitly.
    // This unfortunately means that, for some char* p and size_t s,
    //   BoundedPtr<char> c(p, s); // OK
    //   BoundedPtr<double> d = c; // INVALID
    //   BoundedPtr<double> d2(c); // OK
    //   d2 = c; // OK
    // because the copy-assignment operator also cannot be a template; for the
    // second line to work, we would need to allow implicit conversions. (The
    // template below is simply an assignment operator, and hence can be used
    // only after construction.)
    // Also note that, even if this were not specified as explicit, the various
    // operator[...] non-member functions accepting BoundedPtr<T> types would
    // still never accept a BoundedPtr<U> --- the type deduction only considers
    // exact matches, and does not take into account valid conversions. There is
    // an exception to this rule if the function definition is within the class
    // definition (which is only possible for friend functions, and not done
    // here).
    // This does not apply to member functions, which would accept a
    // BoundedPtr<U> and implicitly convert it to a BoundedPtr<T>, if the
    // below were not specified explicit.
    template <typename U>
    GRACE_HOST_DEVICE explicit BoundedPtr(const BoundedPtr<U>& other);

    // Again, copy-assigment operator must not be a template. We use the default
    // copy-assignment operator.
    template <typename U>
    GRACE_HOST_DEVICE BoundedPtr<T>& operator=(const BoundedPtr<U>& other);

    GRACE_HOST_DEVICE T& operator*();

    GRACE_HOST_DEVICE const T& operator*() const;

    GRACE_HOST_DEVICE T& operator[](difference_type i);

    GRACE_HOST_DEVICE const T& operator[](difference_type i) const;

    GRACE_HOST_DEVICE T* operator->();

    GRACE_HOST_DEVICE const T* operator->() const;

    GRACE_HOST_DEVICE BoundedPtr<T>& operator+=(const difference_type n);

    GRACE_HOST_DEVICE BoundedPtr<T>& operator-=(const difference_type n);

    friend difference_type operator-<T>(const BoundedPtr<T>& lhs,
                                        const BoundedPtr<T>& rhs);
    // Could be implemented from operator<. Not done so for efficiency.
    friend bool operator==<T>(const BoundedPtr<T>& lhs,
                              const BoundedPtr<T>& rhs);
    // It doesn't make sense to swap BoundedPtrs of different value_types.
    friend void swap<T>(BoundedPtr<T>& lhs, BoundedPtr<T>& rhs);

private:
    // Same type so they may safely be compared.
    // begin_ and end_ correspond to the total bounded memory range, and hence
    // are not necessarily aligned for a type T, and should _never_ be
    // dereferenced.
    T* begin_;
    T* end_;
    T* ptr_;

    // Ensures alignment, increasing address when necessary.
    GRACE_HOST_DEVICE void align_to(const size_t alignment);
};

//
// BoundedPtr operators
//

template <typename T>
GRACE_HOST_DEVICE bool operator<(const BoundedPtr<T>& lhs,
                                 const BoundedPtr<T>& rhs);

// Prefix.
template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T>& operator++(BoundedPtr<T>& bptr);

// Postfix.
template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T> operator++(BoundedPtr<T>& bptr, int);

// Prefix.
template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T>& operator--(BoundedPtr<T>& bptr);

// Postfix.
template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T> operator--(BoundedPtr<T>& bptr, int);

template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T> operator+(const BoundedPtr<T>& lhs,
                                          const typename BoundedPtr<T>::difference_type rhs);

template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T> operator+(const typename BoundedPtr<T>::difference_type lhs,
                                          const BoundedPtr<T>& rhs);

template <typename T>
GRACE_HOST_DEVICE BoundedPtr<T> operator-(const BoundedPtr<T>& lhs,
                                          const typename BoundedPtr<T>::difference_type rhs);

template <typename T>
GRACE_HOST_DEVICE bool operator!=(const BoundedPtr<T>& lhs,
                                  const BoundedPtr<T>& rhs);

template <typename T>
GRACE_HOST_DEVICE bool operator>(const BoundedPtr<T>& lhs,
                                 const BoundedPtr<T>& rhs);

template <typename T>
GRACE_HOST_DEVICE bool operator<=(const BoundedPtr<T>& lhs,
                                  const BoundedPtr<T>& rhs);

template <typename T>
GRACE_HOST_DEVICE bool operator>=(const BoundedPtr<T>& lhs,
                                  const BoundedPtr<T>& rhs);

template <typename T>
GRACE_HOST_DEVICE void swap(BoundedPtr<T>& lhs,
                            BoundedPtr<T>& rhs);

} // namespace grace

#include "grace/generic/detail/boundedptr-inl.h"
