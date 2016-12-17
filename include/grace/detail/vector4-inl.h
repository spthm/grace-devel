#pragma once

// No grace/vector.h include.
// This should only ever be included by grace/vector.h.

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
T* Vector<4, T>::data()
{
    return reinterpret_cast<T*>(this);
}

template <typename T>
GRACE_HOST_DEVICE
const T* Vector<4, T>::data() const
{
    return reinterpret_cast<const T*>(this);
}

template <typename T>
GRACE_HOST_DEVICE
T& Vector<4, T>::operator[](size_t i)
{
    return this->data()[i];
}

template <typename T>
GRACE_HOST_DEVICE
const T& Vector<4, T>::operator[](size_t i) const
{
    // Overloads to const data().
    return this->data()[i];
}

template <typename T>
GRACE_HOST_DEVICE
Vector<3, T>& Vector<4, T>::vec3()
{
    return *reinterpret_cast<Vector<3, T>*>(this->data());
}

template <typename T>
GRACE_HOST_DEVICE
const Vector<3, T>& Vector<4, T>::vec3() const
{
    // Overloads to const data().
    return *reinterpret_cast<const Vector<3, T>*>(this->data());
}

} // namespace grace
