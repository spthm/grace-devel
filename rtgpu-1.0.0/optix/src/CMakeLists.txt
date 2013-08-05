fltk_wrap_ui(rtgpu gui.fl)

set(CUDA_64_BIT_DEVICE_CODE OFF)

include_directories(${FLTK_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS}
    ${OpenMesh_INCLUDE_DIRS} ${Boost_INCLUDE_DIR} ${OPTIX_INCLUDE_DIRS}
		    ${CUDA_INCLUDE_DIRS})

CUDA_COMPILE_PTX(PTX_FILES pinhole_camera.cu sphere.cu flat_shading.cu
			   plastic.cu box.cu parallelogram.cu
                           triangle_mesh.cu matte.cu
			   OPTIONS "-arch=sm_13")

add_executable(rtgpu main.cpp canvas.cpp model.cpp util.cpp
		    ${PTX_FILES} ${rtgpu_FLTK_UI_SRCS})

target_link_libraries(rtgpu ${FLTK_LIBRARIES} ${GLEW_LIBRARIES} 
    s3d_util s3d_math rply s3d_image GLU ${OpenMesh_LIBRARIES}
    optixu optix ${CUDA_LIBRARIES})


add_precompiled_header(rtgpu pch.h)
