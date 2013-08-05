find_package(Boost REQUIRED system)
find_package(JPEG)
find_package(PNG)

set(HAS_JPEG ${JPEG_FOUND})
set(HAS_PNG ${PNG_FOUND})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-parentheses")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-parentheses")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Wno-parentheses")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -Wno-parentheses")

add_definitions(-D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE
				-D_LARGEFILE64_SOURCE)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in
	           ${CMAKE_CURRENT_SOURCE_DIR}/config.h ESCAPE_QUOTES)

add_library(s3d_image image.cpp io.cpp convolution.cpp traits.cpp
			kernel.cpp packed_image.cpp laplacian.cpp large_image.cpp
			format/png.cpp format/jpeg.cpp format/bitmap.cpp rescale.cpp
			format/kro.cpp format/format.cpp)
target_link_libraries(s3d_image s3d_color s3d_util ${Boost_SYSTEM_LIBRARY})
include_directories(${Boost_INCLUDE_DIR})
	
if(HAS_JPEG)
	target_link_libraries(s3d_image ${JPEG_LIBRARIES})
	include_directories(${JPEG_INCLUDE_DIR})
endif()

if(HAS_PNG)
	target_link_libraries(s3d_image ${PNG_LIBRARIES})
	include_directories(${PNG_INCLUDE_DIR})
	add_definitions(${PNG_DEFINITIONS})
endif()

add_precompiled_header(s3d_image pch.h)
