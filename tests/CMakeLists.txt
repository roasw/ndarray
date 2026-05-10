add_executable(dlpack_test demo/dlpack_column_major.cpp)
target_link_libraries(dlpack_test ndarray)

add_test(NAME dlpack_column_major COMMAND dlpack_test)

add_executable(ndarray_typed_test ndarray/ndarray_typed.cpp)
target_link_libraries(ndarray_typed_test ndarray)

add_test(NAME ndarray_typed COMMAND ndarray_typed_test)

add_executable(upsample_2d_fourier_runtime_test algorithm/upsample_2d_fourier_runtime.cpp)
target_link_libraries(
    upsample_2d_fourier_runtime_test
    upsample_2d_fourier_algorithm
    ${TORCH_LIBRARIES}
)
target_compile_definitions(
    upsample_2d_fourier_runtime_test
    PRIVATE
        UPSAMPLE_2D_FOURIER_CPU_F32_MODEL_PATH="${UPSAMPLE_2D_FOURIER_CPU_F32_MODEL_PATH}"
        UPSAMPLE_2D_FOURIER_CPU_F64_MODEL_PATH="${UPSAMPLE_2D_FOURIER_CPU_F64_MODEL_PATH}"
)
add_dependencies(upsample_2d_fourier_runtime_test upsample_2d_fourier_model)

add_test(NAME upsample_2d_fourier_runtime COMMAND upsample_2d_fourier_runtime_test)
set_tests_properties(upsample_2d_fourier_runtime PROPERTIES ENVIRONMENT "${PYTHON_FREE_ENV}")
