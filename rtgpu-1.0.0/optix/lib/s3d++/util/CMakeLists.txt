find_package(Boost REQUIRED)

add_library(s3d_util STATIC time.cpp demangle.cpp file.cpp recursion_guard.cpp
	                        visitor.cpp fps.cpp)
include_directories(${Boost_INCLUDE_DIR})

add_precompiled_header(s3d_util pch.h)
