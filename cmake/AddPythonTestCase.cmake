function(add_python_test_case name script)
    add_test(
        NAME ${name}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/${script} ${ARGN}
    )
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "${PYTHON_TEST_ENV}")
endfunction()
