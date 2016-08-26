#pragma once

#include "grace/vector.h"

namespace grace {

template <typename T>
template <typename U>
GRACE_HOST_DEVICE
Vector<3, T>& Vector<3, T>::operator=(const Vector<3, U>& rhs)
{
    Base::operator=(rhs);
    return *this;
}

template <typename T>
GRACE_HOST_DEVICE
T* Vector<3, T>::data()
{
    // Element order in memory guaranteed identical to order of declaration.
    // However, depending on the architecture and compiler, this may be unsafe
    // for sizeof(T) < 4: there may be padding after each element.
    return reinterpret_cast<T*>(this);
}

template <typename T>
GRACE_HOST_DEVICE
const T* Vector<3, T>::data() const
{
    return reinterpret_cast<const T*>(this);
}

template <typename T>
GRACE_HOST_DEVICE
T& Vector<3, T>::operator[](size_t i)
{
    return this->data()[i];
}

template <typename T>
GRACE_HOST_DEVICE
const T& Vector<3, T>::operator[](size_t i) const
{
    // Overloads to const data().
    return this->data()[i];
}

} // namespace grace
