fltk_wrap_ui(rtgpu gui.fl)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/sdk/C/lib)
#                 ${CUDA_TOOLKIT_ROOT_DIR}/sdk/C/common/lib/linux)

include_directories(${FLTK_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS}
    ${OpenMesh_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} 
    ${CUDA_INCLUDE_DIRS} ${OPTIX_INCLUDE_DIRS})
##    ${CUDA_TOOLKIT_ROOT_DIR}/sdk/C/common/inc)

CUDA_COMPILE(CUDA_FILES raytrace.cu cuda_util.cu local_intersections.cu
    packet_decomposition.cu ray_compression.cu cuda_bvh.cu traversal.cu
             OPTIONS -arch sm_13
                     -Xcompiler=-std=c++98
                     -Xptxas=-v
                     -use_fast_math
                     -Xopencc -LIST:source=on)


add_executable(rtgpu main.cpp canvas.cpp model.cpp util.cpp bvh.cpp timer.cpp
    alloc.cpp
    ${CUDA_FILES} ${rtgpu_FLTK_UI_SRCS})

target_link_libraries(rtgpu ${FLTK_LIBRARIES} ${GLEW_LIBRARIES} 
    s3d_util s3d_math rply s3d_image GLU ${OpenMesh_LIBRARIES}
    ${CUDA_LIBRARIES} cudpp_x86_64)


add_precompiled_header(rtgpu pch.h)
