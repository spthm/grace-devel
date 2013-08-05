#ifndef S3D_UTIL_TYPE_TRAITS_FWD_H
#define S3D_UTIL_TYPE_TRAITS_FWD_H

namespace s3d
{

template <class T>
struct is_concept;

template <class T, class EN=void>
struct remove_concept;

template <class T, int N=1>
struct value_type;

template <class T, class U, int N=-1> // -1 -> usa order<T>::value
struct rebind;

}

#endif
