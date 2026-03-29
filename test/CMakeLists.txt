add_executable(dlpack_test dlpack_column_major.cpp)
target_link_libraries(dlpack_test ${TORCH_LIBRARIES} ${ARMADILLO_LIBRARIES})
target_include_directories(dlpack_test PRIVATE ${ARMADILLO_INCLUDE_DIRS})

add_test(NAME dlpack_column_major COMMAND dlpack_test)
