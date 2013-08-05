find_package(Boost REQUIRED)

add_library(s3d_color util.cpp color.cpp)
include_directories(${Boost_INCLUDE_DIR})

add_precompiled_header(s3d_color pch.h)
