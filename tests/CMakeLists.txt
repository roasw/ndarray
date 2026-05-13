#######
# C++ #
#######

add_executable(dlpack_test demo/dlpack_column_major.cpp)
target_link_libraries(dlpack_test PRIVATE ndarray ndarray_project_options)

add_test(NAME dlpack_column_major COMMAND dlpack_test)

add_executable(ndarray_typed_test ndarray/ndarray_typed.cpp)
target_link_libraries(ndarray_typed_test PRIVATE ndarray ndarray_project_options)

add_test(NAME ndarray_typed COMMAND ndarray_typed_test)

add_executable(ndarray_arithmetic_test ndarray/ndarray_arithmetic.cpp)
target_link_libraries(ndarray_arithmetic_test PRIVATE ndarray ndarray_project_options)

add_test(NAME ndarray_arithmetic COMMAND ndarray_arithmetic_test)

add_executable(upsample_2d_fourier_runtime_test algorithm/upsample_2d_fourier_runtime.cpp)
target_link_libraries(
    upsample_2d_fourier_runtime_test
    PRIVATE
        upsample_2d_fourier_algorithm
        ${TORCH_LIBRARIES}
        ndarray_project_options
)
target_compile_definitions(
    upsample_2d_fourier_runtime_test
    PRIVATE
        UPSAMPLE_2D_FOURIER_METADATA_PATH="${UPSAMPLE_2D_FOURIER_METADATA_PATH}"
)
add_dependencies(upsample_2d_fourier_runtime_test upsample_2d_fourier_model)

add_test(NAME upsample_2d_fourier_runtime COMMAND upsample_2d_fourier_runtime_test)

add_executable(
    upsample_2d_fourier_parity_runtime_test
    algorithm/upsample_2d_fourier_parity_runtime.cpp
)
target_link_libraries(
    upsample_2d_fourier_parity_runtime_test
    PRIVATE
        upsample_2d_fourier_algorithm
        upsample_2d_fourier_kernel_algorithm
        ${NDARRAY_TORCH_KERNEL_DSO_TARGET}
        ${TORCH_LIBRARIES}
        ndarray_project_options
)
target_compile_definitions(
    upsample_2d_fourier_parity_runtime_test
    PRIVATE
        UPSAMPLE_2D_FOURIER_METADATA_PATH="${UPSAMPLE_2D_FOURIER_METADATA_PATH}"
        UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH="${UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH}"
)
add_dependencies(
    upsample_2d_fourier_parity_runtime_test
    upsample_2d_fourier_model
    upsample_2d_fourier_kernel_model
)

add_test(
    NAME upsample_2d_fourier_parity_runtime
    COMMAND upsample_2d_fourier_parity_runtime_test
)

add_executable(
    upsample_2d_fourier_kernel_runtime_test
    algorithm/upsample_2d_fourier_kernel_runtime.cpp
)
target_link_libraries(
    upsample_2d_fourier_kernel_runtime_test
    PRIVATE
        upsample_2d_fourier_kernel_algorithm
        ${NDARRAY_TORCH_KERNEL_DSO_TARGET}
        ${TORCH_LIBRARIES}
        ndarray_project_options
)
target_compile_definitions(
    upsample_2d_fourier_kernel_runtime_test
    PRIVATE
        UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH="${UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH}"
)
add_dependencies(
    upsample_2d_fourier_kernel_runtime_test
    upsample_2d_fourier_kernel_model
)

add_test(
    NAME upsample_2d_fourier_kernel_runtime
    COMMAND upsample_2d_fourier_kernel_runtime_test
)

##########
# Python #
##########

set(PYTHON_TEST_ENV
    "PYTHONPATH=${CMAKE_BINARY_DIR}/python:${CMAKE_SOURCE_DIR}/python:$ENV{PYTHONPATH}"
)

add_python_test_case(
    python_ndarray_dlpack_bridge
    tests/python/test_ndarray_dlpack_bridge.py
)
add_python_test_case(
    python_upsample_2d_fourier
    tests/python/test_upsample_2d_fourier.py
    --package-metadata
    ${UPSAMPLE_2D_FOURIER_METADATA_PATH}
)
add_python_test_case(
    python_upsample_2d_fourier_kernel
    tests/python/test_upsample_2d_fourier_kernel.py
    --package-metadata
    ${UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH}
    --kernel-lib
    $<TARGET_FILE:${NDARRAY_TORCH_KERNEL_DSO_TARGET}>
)
add_python_test_case(
    python_kernel_upsample_2d_fourier
    tests/python/test_kernel_upsample_2d_fourier.py
    --kernel-lib
    $<TARGET_FILE:${NDARRAY_TORCH_KERNEL_DSO_TARGET}>
)

add_python_test_case(
    python_upsample_2d_fourier_known_output
    tests/python/test_upsample_2d_fourier_known_output.py
    --package-metadata
    ${UPSAMPLE_2D_FOURIER_METADATA_PATH}
    --kernel-package-metadata
    ${UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH}
    --kernel-lib
    $<TARGET_FILE:${NDARRAY_TORCH_KERNEL_DSO_TARGET}>
    --plots-dir
    ${CMAKE_BINARY_DIR}/plots
)

add_python_test_case(
    python_upsample_2d_fourier_runtime_checks
    tests/python/test_upsample_2d_fourier_runtime_checks.py
    --package-metadata
    ${UPSAMPLE_2D_FOURIER_METADATA_PATH}
)
