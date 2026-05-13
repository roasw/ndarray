cmake_minimum_required(VERSION 3.15)
project(ndarray LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
find_package(Armadillo REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Doxygen QUIET)

include(cmake/AOTICompile.cmake)
include(cmake/AddAlgorithmLibrary.cmake)
include(cmake/AddTorchKernelLibrary.cmake)
include(cmake/AddPythonTestCase.cmake)

add_library(ndarray_project_options INTERFACE)
target_compile_options(
    ndarray_project_options
    INTERFACE
        $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang,GNU>:-Wall;-Wextra;-Wpedantic;-Werror>
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/W4;/WX>
)
if(TORCH_CXX_FLAGS)
    separate_arguments(NDARRAY_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(
        ndarray_project_options
        INTERFACE
            ${NDARRAY_TORCH_CXX_FLAGS}
    )
endif()

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
target_link_libraries(ndarray PRIVATE ndarray_project_options)

add_algorithm_library(
    upsample_2d_fourier_algorithm
    src/algorithm/upsample_2d_fourier.cpp
)
target_link_libraries(upsample_2d_fourier_algorithm PRIVATE ndarray_project_options)
add_algorithm_library(
    upsample_2d_fourier_kernel_algorithm
    src/algorithm/upsample_2d_fourier_kernel.cpp
)
target_link_libraries(
    upsample_2d_fourier_kernel_algorithm
    PRIVATE
        ndarray_project_options
)

set(NDARRAY_TORCH_KERNEL_DSO_TARGET ndarray_torch_kernels)
add_torch_kernel_library(
    ${NDARRAY_TORCH_KERNEL_DSO_TARGET}
    src/kernel/upsample_2d_fourier.cpp
)
target_link_libraries(
    ${NDARRAY_TORCH_KERNEL_DSO_TARGET}
    PRIVATE
        ndarray_project_options
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
target_link_libraries(_ndarray PRIVATE ndarray_project_options)
set_target_properties(_ndarray PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)

################
# AOTI exports #
################

add_aoti_compile_target(
    TARGET upsample_2d_fourier_model
    ALGORITHM_FILE python/ndarry/algorithm/upsample_2d_fourier.py
    METADATA_VAR UPSAMPLE_2D_FOURIER_METADATA_PATH
    MODE ${CMAKE_BUILD_TYPE}
    CONFIG max_factor=8
)

add_aoti_compile_target(
    TARGET upsample_2d_fourier_kernel_model
    ALGORITHM_FILE python/ndarry/algorithm/upsample_2d_fourier_kernel.py
    METADATA_VAR UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH
    MODE ${CMAKE_BUILD_TYPE}
    CONFIG max_factor=8
    CONFIG kernel_lib=$<TARGET_FILE:${NDARRAY_TORCH_KERNEL_DSO_TARGET}>
    DEPENDS ${NDARRAY_TORCH_KERNEL_DSO_TARGET}
)

##############
# Test cases #
##############

enable_testing()
add_subdirectory(tests)

########################
# Documentation target #
########################

if(DOXYGEN_FOUND)
    set(NDARRAY_DOXYFILE_IN ${CMAKE_SOURCE_DIR}/cmake/Doxyfile.in)
    set(NDARRAY_DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    set(NDARRAY_DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
    configure_file(${NDARRAY_DOXYFILE_IN} ${NDARRAY_DOXYFILE} @ONLY)

    add_custom_target(
        doxygen
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NDARRAY_DOXYGEN_OUTPUT_DIR}
        COMMAND ${DOXYGEN_EXECUTABLE} ${NDARRAY_DOXYFILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
else()
    message(STATUS "Doxygen not found; 'doxygen' target is unavailable")
endif()
