add_executable(dlpack_test dlpack_column_major.cpp)
target_link_libraries(dlpack_test ndarray_lib)

add_test(NAME dlpack_column_major COMMAND dlpack_test)

add_executable(ndarray_float_test ndarray_float.cpp)
target_link_libraries(ndarray_float_test ndarray_lib)

add_test(NAME ndarray_float COMMAND ndarray_float_test)
