cmake_minimum_required(VERSION 3.9)

if (POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

project(make-dual-and-bricks VERSION 0.0.1 LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(submodules/umesh EXCLUDE_FROM_ALL)
include_directories(/usr/local/cuda-12.2/include)
set(TBB_INCLUDE_DIR "/usr/include/tbb")

# ==================================================================
add_executable(amrMakeDualMesh
  makeDual.cpp
  )

target_link_libraries(amrMakeDualMesh
  PUBLIC
  umesh
  )

# ==================================================================
add_executable(amrMakeGrids
  makeGrids.cpp
  )

target_link_libraries(amrMakeGrids
  PUBLIC
  umesh
  )

# ==================================================================
add_executable(amrMakeGrids_cuda3
  makeGrids3Kernels.cu
  )

target_link_libraries(amrMakeGrids_cuda3
  PUBLIC
  umesh
  )

# ==================================================================
add_executable(amrMakeGrids_cuda4
  makeGrids4Kernels.cu
  )

target_link_libraries(amrMakeGrids_cuda4
  PUBLIC
  umesh
  )


set_target_properties(amrMakeGrids_cuda3 PROPERTIES CUDA_ARCHITECTURES "75")
