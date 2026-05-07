add_executable(dlpack_test demo/dlpack_column_major.cpp)
target_link_libraries(dlpack_test ndarray)

add_test(NAME dlpack_column_major COMMAND dlpack_test)

add_executable(ndarray_float_test ndarray/ndarray_float.cpp)
target_link_libraries(ndarray_float_test ndarray)

add_test(NAME ndarray_float COMMAND ndarray_float_test)

add_executable(upsample_pt2_runtime_test algorithm/upsample_pt2_runtime.cpp)
target_link_libraries(upsample_pt2_runtime_test upsample_algorithm ${TORCH_LIBRARIES})
target_compile_definitions(upsample_pt2_runtime_test PRIVATE UPSAMPLE_PT2_PATH="${UPSAMPLE_PT2_PATH}")
add_dependencies(upsample_pt2_runtime_test upsample_pt2)

add_test(NAME upsample_pt2_runtime COMMAND upsample_pt2_runtime_test)
set_tests_properties(upsample_pt2_runtime PROPERTIES ENVIRONMENT "${PYTHON_FREE_ENV}")
