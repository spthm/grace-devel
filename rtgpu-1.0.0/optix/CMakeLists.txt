# cmake-2.6.0 nao consegue linkar o boost_system
cmake_minimum_required(VERSION 2.6.1 FATAL_ERROR)

project(rtgpu CXX C)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/etc)

if(CMAKE_COMPILER_IS_GNUCXX)
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG(-std=c++0x HAS_CXX0X)
    if(NOT HAS_CXX0X)
		message(FATAL_ERROR "I need a C++0x-capable compiler")
	endif()

	set(CMAKE_REQUIRED_FLAGS "-std=gnu++0x")

	set(BASE_FLAGS "-std=gnu++0x -Wno-deprecated")
	set(RELEASE_FLAGS "-O3 -march=core2 -msse3 -mfpmath=both -ffast-math -DNDEBUG -ffast-math -Wno-parentheses")
	set(DEBUG_FLAGS "-O0 -ggdb")

	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${BASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE 
		"${CMAKE_CXX_FLAGS_RELEASE} ${BASE_FLAGS} ${RELEASE_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG 
		"${CMAKE_CXX_FLAGS_DEBUG} ${BASE_FLAGS} ${DEBUG_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO 
		"${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${BASE_FLAGS} ${DEBUG_FLAGS} ${RELEASE_FLAGS}")

    set(CMAKE_DEBUG_POSTFIX "d")
else()
	message(FATAL_ERROR "Compiler not supported, try gcc>=4.5")
endif()

include(PCHSupport)

set(Boost_ADDITIONAL_VERSIONS 1.44)

find_package(FLTK REQUIRED)
find_package(Glew REQUIRED)
find_package(Boost REQUIRED)
find_package(CUDA REQUIRED)

find_path(OPTIX_INCLUDE_DIRS optix_math.h PATH_SUFFIXES optix)
if(NOT OPTIX_INCLUDE_DIRS)
	message(FATAL "OptiX must be istalled")
endif()

include_directories(lib)

add_subdirectory(lib EXCLUDE_FROM_ALL)
add_subdirectory(src)
