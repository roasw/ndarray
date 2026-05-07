cmake_minimum_required(VERSION 3.15)
project(ndarray LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

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

find_package(Torch REQUIRED)
find_package(Armadillo REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(UPSAMPLE_PT2_PATH "${CMAKE_BINARY_DIR}/artifacts/upsample_model.pt2")

add_library(ndarray SHARED src/container/ndarray_float.cpp)
target_include_directories(
    ndarray
    PUBLIC
        inc
        ${TORCH_INCLUDE_DIRS}
        ${ARMADILLO_INCLUDE_DIRS}
)
target_link_libraries(ndarray PUBLIC ${TORCH_LIBRARIES} ${ARMADILLO_LIBRARIES})

add_library(upsample_algorithm SHARED src/algorithm/upsample_pt2.cpp)
target_include_directories(
    upsample_algorithm
    PUBLIC
        inc
        ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(upsample_algorithm PUBLIC ndarray ${TORCH_LIBRARIES})

add_custom_command(
    OUTPUT ${UPSAMPLE_PT2_PATH}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/artifacts
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/algorithm/export_upsample_pt2.py --output ${UPSAMPLE_PT2_PATH}
    DEPENDS ${CMAKE_SOURCE_DIR}/python/algorithm/export_upsample_pt2.py
    VERBATIM
)
add_custom_target(upsample_pt2 ALL DEPENDS ${UPSAMPLE_PT2_PATH})

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

enable_testing()
add_subdirectory(tests)

add_test(
    NAME python_ndarray_dlpack_bridge
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/tests/test_ndarray_dlpack_bridge.py
)
add_test(
    NAME python_upsample_pt2
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/tests/test_upsample_pt2.py --package ${UPSAMPLE_PT2_PATH}
)
