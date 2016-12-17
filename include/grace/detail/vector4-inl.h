#pragma once

// No grace/vector.h include.
// This should only ever be included by grace/vector.h.
#include <grace/error.h>

namespace grace {

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<4, T>& Vector<4, T>::operator=(const Vector<4, U>& rhs)
{
    Base::operator=(rhs);
    return *this;
}

template <typename T>
GRACE_HOST_DEVICE
T& Vector<4, T>::operator[](int i)
{
    switch (i) {
        case 0: return this->x;
        case 1: return this->y;
        case 2: return this->z;
        case 3: return this->w;
    }
    GRACE_ASSERT(0, vector4_invalid_index_access);
}

template <typename T>
GRACE_HOST_DEVICE
const T& Vector<4, T>::operator[](int i) const
{
    switch (i) {
        case 0: return this->x;
        case 1: return this->y;
        case 2: return this->z;
        case 3: return this->w;
    }
    GRACE_ASSERT(0, vector4_invalid_index_access);
}

} // namespace grace
