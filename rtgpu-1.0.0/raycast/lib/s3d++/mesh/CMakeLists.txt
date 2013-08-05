find_package(Boost REQUIRED)

add_library(s3d_mesh triangle.cpp triangle_adjacency.cpp)
include_directories(${Boost_INCLUDE_DIR})
