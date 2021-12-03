cmake_minimum_required(VERSION 3.19)

project(05_hetero LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)
find_package(OpenCL 2.2 REQUIRED)

set(TARGET_NAME hetero)

file(GLOB_RECURSE TARGET_HEADERS include/*.hpp)
file(GLOB_RECURSE TARGET_SRC src/*.cpp)

add_executable(${TARGET_NAME} ${TARGET_HEADERS} ${TARGET_SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE "CL_TARGET_OPENCL_VERSION=220")
target_compile_definitions(${TARGET_NAME} PRIVATE "KERNELS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/kernels/\"")
target_include_directories(${TARGET_NAME} PRIVATE include)
target_link_libraries(${TARGET_NAME} PUBLIC OpenMP::OpenMP_CXX PRIVATE OpenCL::OpenCL)
