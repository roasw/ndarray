#######
# C++ #
#######

add_cpp_test_case(
    TARGET dlpack_test
    SOURCE demo/dlpack_column_major.cpp
    TEST_NAME dlpack_column_major
    LINK_LIBRARIES ndarray
)

add_cpp_test_case(
    TARGET ndarray_typed_test
    SOURCE ndarray/ndarray_typed.cpp
    TEST_NAME ndarray_typed
    LINK_LIBRARIES ndarray
)

add_cpp_test_case(
    TARGET ndarray_arithmetic_test
    SOURCE ndarray/ndarray_arithmetic.cpp
    TEST_NAME ndarray_arithmetic
    LINK_LIBRARIES ndarray
)

add_cpp_test_case(
    TARGET upsample_2d_fourier_runtime_test
    SOURCE algorithm/upsample_2d_fourier_runtime.cpp
    TEST_NAME upsample_2d_fourier_runtime
    LINK_LIBRARIES
        upsample_2d_fourier_algorithm
        ${TORCH_LIBRARIES}
    COMPILE_DEFINITIONS
        UPSAMPLE_2D_FOURIER_METADATA_PATH="${UPSAMPLE_2D_FOURIER_METADATA_PATH}"
    DEPENDS
        upsample_2d_fourier_model
)

add_cpp_test_case(
    TARGET upsample_2d_fourier_parity_runtime_test
    SOURCE algorithm/upsample_2d_fourier_parity_runtime.cpp
    TEST_NAME upsample_2d_fourier_parity_runtime
    LINK_LIBRARIES
        upsample_2d_fourier_algorithm
        upsample_2d_fourier_kernel_algorithm
        ${NDARRAY_TORCH_KERNEL_DSO_TARGET}
        ${TORCH_LIBRARIES}
    COMPILE_DEFINITIONS
        UPSAMPLE_2D_FOURIER_METADATA_PATH="${UPSAMPLE_2D_FOURIER_METADATA_PATH}"
        UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH="${UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH}"
    DEPENDS
        upsample_2d_fourier_model
        upsample_2d_fourier_kernel_model
)

add_cpp_test_case(
    TARGET upsample_2d_fourier_kernel_runtime_test
    SOURCE algorithm/upsample_2d_fourier_kernel_runtime.cpp
    TEST_NAME upsample_2d_fourier_kernel_runtime
    LINK_LIBRARIES
        upsample_2d_fourier_kernel_algorithm
        ${NDARRAY_TORCH_KERNEL_DSO_TARGET}
        ${TORCH_LIBRARIES}
    COMPILE_DEFINITIONS
        UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH="${UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH}"
    DEPENDS
        upsample_2d_fourier_kernel_model
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
