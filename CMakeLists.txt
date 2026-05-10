cmake_minimum_required(VERSION 3.15)
project(ndarray LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(NOT Torch_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE TORCH_CMAKE_QUERY_STATUS
    )

    if(TORCH_CMAKE_QUERY_STATUS EQUAL 0 AND TORCH_CMAKE_PREFIX_PATH)
        list(PREPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
    endif()
endif()

if(NOT pybind11_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_CMAKE_QUERY_STATUS
    )

    if(PYBIND11_CMAKE_QUERY_STATUS EQUAL 0 AND PYBIND11_CMAKE_DIR)
        list(PREPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
    endif()
endif()

find_package(Torch REQUIRED)
find_package(Armadillo REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

include(cmake/AOTICompile.cmake)
include(cmake/AddAlgorithmLibrary.cmake)
include(cmake/AddTorchKernelLibrary.cmake)
include(cmake/AddPythonTestCase.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(UPSAMPLE_2D_FOURIER_CPU_F32_MODEL_PATH
    "${CMAKE_BINARY_DIR}/artifacts/upsample_2d_fourier_cpu_f32_model.pt2"
)
set(UPSAMPLE_2D_FOURIER_CPU_F64_MODEL_PATH
    "${CMAKE_BINARY_DIR}/artifacts/upsample_2d_fourier_cpu_f64_model.pt2"
)
set(UPSAMPLE_2D_FOURIER_KERNEL_CPU_F32_MODEL_PATH
    "${CMAKE_BINARY_DIR}/artifacts/upsample_2d_fourier_kernel_cpu_f32_model.pt2"
)
set(UPSAMPLE_2D_FOURIER_KERNEL_CPU_F64_MODEL_PATH
    "${CMAKE_BINARY_DIR}/artifacts/upsample_2d_fourier_kernel_cpu_f64_model.pt2"
)

#################
# Build targets #
#################

add_library(ndarray SHARED src/container/ndarray_typed.cpp)
target_include_directories(
    ndarray
    PUBLIC
        inc
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
add_torch_kernel_library(
    upsample_2d_fourier_cpu_kernel
    src/kernel/upsample_2d_fourier_cpu.cpp
)

pybind11_add_module(_ndarray MODULE src/python/py_ndarray.cpp)
target_include_directories(
    _ndarray
    PRIVATE
        inc
        ${TORCH_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
)
target_link_libraries(_ndarray PRIVATE ndarray ${TORCH_LIBRARIES})
set_target_properties(_ndarray PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)

################
# AOTI exports #
################

add_aoti_compile_target(
    TARGET upsample_2d_fourier_model
    ALGORITHM_MODULE python.algorithm.upsample_2d_fourier
    ALGORITHM_CLASS Upsample2DFourier
    OUTPUT_DIR ${CMAKE_BINARY_DIR}/artifacts
    MODE ${CMAKE_BUILD_TYPE}
    OUTPUTS
        ${UPSAMPLE_2D_FOURIER_CPU_F32_MODEL_PATH}
        ${UPSAMPLE_2D_FOURIER_CPU_F64_MODEL_PATH}
    CONFIG max_factor=8
    DEPENDS ${CMAKE_SOURCE_DIR}/python/algorithm/upsample_2d_fourier.py
)

add_aoti_compile_target(
    TARGET upsample_2d_fourier_kernel_model
    ALGORITHM_MODULE python.algorithm.upsample_2d_fourier_kernel
    ALGORITHM_CLASS Upsample2DFourierKernel
    OUTPUT_DIR ${CMAKE_BINARY_DIR}/artifacts
    MODE ${CMAKE_BUILD_TYPE}
    OUTPUTS
        ${UPSAMPLE_2D_FOURIER_KERNEL_CPU_F32_MODEL_PATH}
        ${UPSAMPLE_2D_FOURIER_KERNEL_CPU_F64_MODEL_PATH}
    CONFIG max_factor=8
    CONFIG kernel_lib=$<TARGET_FILE:upsample_2d_fourier_cpu_kernel>
    DEPENDS
        ${CMAKE_SOURCE_DIR}/python/algorithm/upsample_2d_fourier_kernel.py
        upsample_2d_fourier_cpu_kernel
)

##############
# Test cases #
##############

set(PYTHON_RUNTIME_LIB_DIR "")
get_filename_component(PYTHON3_LIB_PARENT_DIR "${Python3_LIBRARIES}" DIRECTORY)
if(PYTHON3_LIB_PARENT_DIR)
    set(PYTHON_RUNTIME_LIB_DIR "${PYTHON3_LIB_PARENT_DIR}")
endif()

set(PYTHON_FREE_ENV
    "PYTHONHOME="
    "PYTHONPATH="
    "PYTHONSTARTUP="
    "PYTHONUSERBASE="
    "PYTHONNOUSERSITE=1"
)
if(PYTHON_RUNTIME_LIB_DIR)
    list(APPEND PYTHON_FREE_ENV "DYLD_LIBRARY_PATH=${PYTHON_RUNTIME_LIB_DIR}")
endif()

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
    --package-f32
    ${UPSAMPLE_2D_FOURIER_CPU_F32_MODEL_PATH}
    --package-f64
    ${UPSAMPLE_2D_FOURIER_CPU_F64_MODEL_PATH}
)
add_python_test_case(
    python_upsample_2d_fourier_kernel
    python/tests/test_upsample_2d_fourier_kernel.py
    --package-f32
    ${UPSAMPLE_2D_FOURIER_KERNEL_CPU_F32_MODEL_PATH}
    --package-f64
    ${UPSAMPLE_2D_FOURIER_KERNEL_CPU_F64_MODEL_PATH}
    --kernel-lib
    $<TARGET_FILE:upsample_2d_fourier_cpu_kernel>
)
add_python_test_case(
    python_kernel_upsample_2d_fourier
    python/tests/test_kernel_upsample_2d_fourier.py
    --kernel-lib
    $<TARGET_FILE:upsample_2d_fourier_cpu_kernel>
)
