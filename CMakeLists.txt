# C++ implementation placeholder (PointNet++ CUDA kernels, etc.)
cmake_minimum_required(VERSION 3.10)
project(Frustum3D-Lite)
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native -Wall -Wextra")


find_package(PCL 1.12 REQUIRED COMPONENTS common io filters segmentation search kdtree)
find_package(Eigen3 3.3 REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "89")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) 

include_directories(${PCL_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS})

add_library(dynamic_frustum STATIC
    src/DynamicFrustum.cpp
    src/tools/priorCrop.cpp
)
target_include_directories(dynamic_frustum PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(
    dynamic_frustum
    ${PCL_LIBRARIES}
    ${CMAKE_THREAD_LIBS_INIT}
)

# ── Test: priorCrop 裁剪效果验证 ──
add_executable(test_prior_crop
    src/test/test_prior_crop.cpp
)
target_include_directories(test_prior_crop PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(test_prior_crop
    dynamic_frustum
    ${PCL_LIBRARIES}
)

# ── C-wrapper shared library (for Python ctypes) ──
add_library(frustum_preprocess SHARED
    src/tools/c_wrapper.cpp
    src/tools/priorCrop.cpp
    src/DynamicFrustum.cpp
)
target_include_directories(frustum_preprocess PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(frustum_preprocess
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
)
set_target_properties(frustum_preprocess PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# ── Batch preprocess library (OpenMP parallel) ──
add_library(batch_preprocess STATIC
    src/tools/batch_preprocess.cpp
)
target_include_directories(batch_preprocess PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(batch_preprocess
    dynamic_frustum
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# ── Benchmark: serial vs parallel speedup ──
add_executable(bench_parallel
    src/test/bench_parallel.cpp
)
target_include_directories(bench_parallel PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(bench_parallel
    batch_preprocess
    dynamic_frustum
    ${PCL_LIBRARIES}
    OpenMP::OpenMP_CXX
)