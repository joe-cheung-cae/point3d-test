cmake_minimum_required(VERSION 3.10)
project(point3d-test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/usr/local/cuda-12.3/include)

find_package(OpenMP REQUIRED)
add_definitions(-D_USE_MATH_DEFINES)

add_executable(point3d_demo main.cpp)
target_link_libraries(point3d_demo OpenMP::OpenMP_CXX)