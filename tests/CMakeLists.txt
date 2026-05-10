add_executable(dlpack_test demo/dlpack_column_major.cpp)
target_link_libraries(dlpack_test ndarray)

add_test(NAME dlpack_column_major COMMAND dlpack_test)

add_executable(ndarray_float_test ndarray/ndarray_float.cpp)
target_link_libraries(ndarray_float_test ndarray)

add_test(NAME ndarray_float COMMAND ndarray_float_test)

add_executable(upsample_runtime_test algorithm/upsample_runtime.cpp)
target_link_libraries(upsample_runtime_test upsample_algorithm ${TORCH_LIBRARIES})
target_compile_definitions(upsample_runtime_test PRIVATE UPSAMPLE_MODEL_PATH="${UPSAMPLE_MODEL_PATH}")
add_dependencies(upsample_runtime_test upsample_model)

add_test(NAME upsample_runtime COMMAND upsample_runtime_test)
set_tests_properties(upsample_runtime PROPERTIES ENVIRONMENT "${PYTHON_FREE_ENV}")
