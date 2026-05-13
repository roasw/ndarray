function(add_cpp_test_case)
    set(options)
    set(one_value_args TARGET SOURCE TEST_NAME)
    set(multi_value_args LINK_LIBRARIES COMPILE_DEFINITIONS DEPENDS)
    cmake_parse_arguments(CPP_TEST "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT CPP_TEST_TARGET)
        message(FATAL_ERROR "add_cpp_test_case requires TARGET")
    endif()
    if(NOT CPP_TEST_SOURCE)
        message(FATAL_ERROR "add_cpp_test_case requires SOURCE")
    endif()
    if(NOT CPP_TEST_TEST_NAME)
        set(CPP_TEST_TEST_NAME "${CPP_TEST_TARGET}")
    endif()

    add_executable(${CPP_TEST_TARGET} ${CPP_TEST_SOURCE})
    target_link_libraries(
        ${CPP_TEST_TARGET}
        PRIVATE
            ${CPP_TEST_LINK_LIBRARIES}
            ndarray_project_options
    )

    if(CPP_TEST_COMPILE_DEFINITIONS)
        target_compile_definitions(
            ${CPP_TEST_TARGET}
            PRIVATE
                ${CPP_TEST_COMPILE_DEFINITIONS}
        )
    endif()

    if(CPP_TEST_DEPENDS)
        add_dependencies(${CPP_TEST_TARGET} ${CPP_TEST_DEPENDS})
    endif()

    add_test(NAME ${CPP_TEST_TEST_NAME} COMMAND ${CPP_TEST_TARGET})
endfunction()
