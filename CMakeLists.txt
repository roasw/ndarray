cmake_minimum_required(VERSION 3.15)
project(ndarray LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)
find_package(Armadillo REQUIRED)

add_library(ndarray_lib src/ndarray_float.cpp)
target_include_directories(ndarray_lib PUBLIC inc)
target_link_libraries(ndarray_lib PUBLIC ${TORCH_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_include_directories(ndarray_lib PUBLIC ${ARMADILLO_INCLUDE_DIRS})

enable_testing()
add_subdirectory(test)
