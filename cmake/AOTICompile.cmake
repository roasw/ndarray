function(add_aoti_compile_target)
    set(options)
    set(one_value_args TARGET ALGORITHM_MODULE ALGORITHM_CLASS OUTPUT_DIR MODE)
    set(multi_value_args CONFIG OUTPUTS DEPENDS)
    cmake_parse_arguments(AOTI "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT AOTI_TARGET)
        message(FATAL_ERROR "add_aoti_compile_target requires TARGET")
    endif()
    if(NOT AOTI_ALGORITHM_MODULE)
        message(FATAL_ERROR "add_aoti_compile_target requires ALGORITHM_MODULE")
    endif()
    if(NOT AOTI_ALGORITHM_CLASS)
        message(FATAL_ERROR "add_aoti_compile_target requires ALGORITHM_CLASS")
    endif()
    if(NOT AOTI_OUTPUT_DIR)
        message(FATAL_ERROR "add_aoti_compile_target requires OUTPUT_DIR")
    endif()
    if(NOT AOTI_OUTPUTS)
        message(FATAL_ERROR "add_aoti_compile_target requires OUTPUTS")
    endif()

    if(NOT AOTI_MODE)
        set(AOTI_MODE debug)
    endif()

    set(_config_args "")
    foreach(_cfg IN LISTS AOTI_CONFIG)
        list(APPEND _config_args --config ${_cfg})
    endforeach()

    add_custom_command(
        OUTPUT ${AOTI_OUTPUTS}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${AOTI_OUTPUT_DIR}
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/aoti-compile.py
            --algorithm-module ${AOTI_ALGORITHM_MODULE}
            --algorithm-class ${AOTI_ALGORITHM_CLASS}
            --output-dir ${AOTI_OUTPUT_DIR}
            --mode ${AOTI_MODE}
            ${_config_args}
        DEPENDS
            ${CMAKE_SOURCE_DIR}/tools/aoti-compile.py
            ${AOTI_DEPENDS}
        VERBATIM
    )

    add_custom_target(${AOTI_TARGET} ALL DEPENDS ${AOTI_OUTPUTS})
endfunction()
