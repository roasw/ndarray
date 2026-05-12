cmake_minimum_required(VERSION 3.15)
project(ndarray LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic -Werror)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
find_package(Armadillo REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include(cmake/AOTICompile.cmake)
include(cmake/AddAlgorithmLibrary.cmake)
include(cmake/AddTorchKernelLibrary.cmake)
include(cmake/AddPythonTestCase.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

#################
# Build targets #
#################

add_library(ndarray SHARED src/container/ndarray_typed.cpp)
target_include_directories(
    ndarray
    PUBLIC
        ${PROJECT_SOURCE_DIR}/inc
        ${TORCH_INCLUDE_DIRS}
        ${ARMADILLO_INCLUDE_DIRS}
)
target_link_libraries(ndarray PUBLIC ${TORCH_LIBRARIES} ${ARMADILLO_LIBRARIES})

add_algorithm_library(
    upsample_2d_fourier_algorithm
    src/algorithm/upsample_2d_fourier.cpp
)
add_algorithm_library(
    upsample_2d_fourier_kernel_algorithm
    src/algorithm/upsample_2d_fourier_kernel.cpp
)

set(NDARRAY_TORCH_KERNEL_DSO_TARGET ndarray_torch_kernels)
add_torch_kernel_library(
    ${NDARRAY_TORCH_KERNEL_DSO_TARGET}
    src/kernel/upsample_2d_fourier_cpu.cpp
)

pybind11_add_module(_ndarray MODULE src/python/py_ndarray.cpp)
target_include_directories(
    _ndarray
    PRIVATE
        ${PROJECT_SOURCE_DIR}/inc
        ${TORCH_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)
target_link_libraries(_ndarray PRIVATE ndarray ${TORCH_LIBRARIES} torch_python)
set_target_properties(_ndarray PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)

################
# AOTI exports #
################

add_aoti_compile_target(
    TARGET upsample_2d_fourier_model
    ALGORITHM_FILE python/algorithm/upsample_2d_fourier.py
    METADATA_VAR UPSAMPLE_2D_FOURIER_METADATA_PATH
    MODE ${CMAKE_BUILD_TYPE}
    CONFIG max_factor=8
)

add_aoti_compile_target(
    TARGET upsample_2d_fourier_kernel_model
    ALGORITHM_FILE python/algorithm/upsample_2d_fourier_kernel.py
    METADATA_VAR UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH
    MODE ${CMAKE_BUILD_TYPE}
    CONFIG max_factor=8
    CONFIG kernel_lib=$<TARGET_FILE:${NDARRAY_TORCH_KERNEL_DSO_TARGET}>
    DEPENDS ${NDARRAY_TORCH_KERNEL_DSO_TARGET}
)

##############
# Test cases #
##############

set(PYTHON_TEST_ENV
    "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${CMAKE_SOURCE_DIR}/python:$ENV{PYTHONPATH}"
)

enable_testing()
add_subdirectory(tests)

add_python_test_case(
    python_ndarray_dlpack_bridge
    python/tests/test_ndarray_dlpack_bridge.py
)
add_python_test_case(
    python_upsample_2d_fourier
    python/tests/test_upsample_2d_fourier.py
    --package-metadata
    ${UPSAMPLE_2D_FOURIER_METADATA_PATH}
)
add_python_test_case(
    python_upsample_2d_fourier_kernel
    python/tests/test_upsample_2d_fourier_kernel.py
    --package-metadata
    ${UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH}
    --kernel-lib
    $<TARGET_FILE:${NDARRAY_TORCH_KERNEL_DSO_TARGET}>
)
add_python_test_case(
    python_kernel_upsample_2d_fourier
    python/tests/test_kernel_upsample_2d_fourier.py
    --kernel-lib
    $<TARGET_FILE:${NDARRAY_TORCH_KERNEL_DSO_TARGET}>
)
