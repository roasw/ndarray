##############
# Benchmarks #
##############

add_python_test_case(
    benchmark_upsample_fourier
    benchmark/benchmark_upsample_fourier.py
    --package-metadata
    ${UPSAMPLE_2D_FOURIER_METADATA_PATH}
    --kernel-package-metadata
    ${UPSAMPLE_2D_FOURIER_KERNEL_METADATA_PATH}
    --kernel-lib
    $<TARGET_FILE:${NDARRAY_TORCH_KERNEL_DSO_TARGET}>
)
set_tests_properties(
    benchmark_upsample_fourier
    PROPERTIES
        LABELS "benchmark"
)
