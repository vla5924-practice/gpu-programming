cmake_minimum_required(VERSION 3.19)

project(01_hello LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL 2.2 REQUIRED)

set(TARGET_NAME hello)

file(GLOB_RECURSE TARGET_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)
file(GLOB_RECURSE TARGET_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(${TARGET_NAME} ${TARGET_HEADERS} ${TARGET_SRC})

target_compile_definitions(${TARGET_NAME} PRIVATE "CL_TARGET_OPENCL_VERSION=220")
target_compile_definitions(${TARGET_NAME} PRIVATE "KERNELS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/kernels/\"")
target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${TARGET_NAME} PRIVATE OpenCL::OpenCL)
