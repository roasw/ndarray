function(add_aoti_compile_target)
    set(options)
    set(one_value_args
        TARGET
        ALGORITHM_FILE
        ALGORITHM_MODULE
        ALGORITHM_CLASS
        OUTPUT_DIR
        MODE
        METADATA_PATH
        METADATA_VAR
    )
    set(multi_value_args CONFIG OUTPUTS DEPENDS)
    cmake_parse_arguments(AOTI "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

    if(NOT AOTI_TARGET)
        message(FATAL_ERROR "add_aoti_compile_target requires TARGET")
    endif()
    if(NOT AOTI_ALGORITHM_MODULE AND NOT AOTI_ALGORITHM_FILE)
        message(FATAL_ERROR "add_aoti_compile_target requires ALGORITHM_MODULE or ALGORITHM_FILE")
    endif()
    if(NOT AOTI_OUTPUT_DIR)
        set(AOTI_OUTPUT_DIR "${CMAKE_BINARY_DIR}/artifacts")
    endif()
    if(NOT AOTI_MODE)
        set(AOTI_MODE debug)
    endif()

    if(AOTI_ALGORITHM_FILE)
        if(IS_ABSOLUTE "${AOTI_ALGORITHM_FILE}")
            set(_algorithm_file_abs "${AOTI_ALGORITHM_FILE}")
        else()
            set(_algorithm_file_abs "${CMAKE_SOURCE_DIR}/${AOTI_ALGORITHM_FILE}")
        endif()

        file(RELATIVE_PATH _algorithm_rel_path "${CMAKE_SOURCE_DIR}" "${_algorithm_file_abs}")
        if(_algorithm_rel_path MATCHES "^\\.\\.")
            message(FATAL_ERROR "ALGORITHM_FILE must be under CMAKE_SOURCE_DIR: ${AOTI_ALGORITHM_FILE}")
        endif()

        set(_algorithm_module_from_file "${_algorithm_rel_path}")
        string(REGEX REPLACE "\\.py$" "" _algorithm_module_from_file "${_algorithm_module_from_file}")
        if(_algorithm_module_from_file STREQUAL _algorithm_rel_path)
            message(FATAL_ERROR "ALGORITHM_FILE must point to a .py file: ${AOTI_ALGORITHM_FILE}")
        endif()
        string(REPLACE "/" "." _algorithm_module_from_file "${_algorithm_module_from_file}")
        string(REPLACE "\\\\" "." _algorithm_module_from_file "${_algorithm_module_from_file}")
    endif()

    if(AOTI_ALGORITHM_MODULE)
        set(_aoti_algorithm_module "${AOTI_ALGORITHM_MODULE}")
        if(AOTI_ALGORITHM_FILE AND NOT _aoti_algorithm_module STREQUAL _algorithm_module_from_file)
            message(FATAL_ERROR
                "ALGORITHM_MODULE (${AOTI_ALGORITHM_MODULE}) does not match module derived "
                "from ALGORITHM_FILE (${_algorithm_module_from_file})"
            )
        endif()
    else()
        set(_aoti_algorithm_module "${_algorithm_module_from_file}")
    endif()

    if(AOTI_METADATA_PATH)
        set(_aoti_metadata_path "${AOTI_METADATA_PATH}")
    else()
        string(REPLACE "." ";" _module_parts "${_aoti_algorithm_module}")
        list(GET _module_parts -1 _algorithm_name)
        set(_aoti_metadata_path "${AOTI_OUTPUT_DIR}/${_algorithm_name}.txt")
    endif()

    if(AOTI_METADATA_VAR)
        set(${AOTI_METADATA_VAR} "${_aoti_metadata_path}" PARENT_SCOPE)
    endif()

    set(_config_args "")
    foreach(_cfg IN LISTS AOTI_CONFIG)
        list(APPEND _config_args --config ${_cfg})
    endforeach()

    set(_algorithm_class_args "")
    if(AOTI_ALGORITHM_CLASS)
        list(APPEND _algorithm_class_args --algorithm-class ${AOTI_ALGORITHM_CLASS})
    endif()

    set(_aoti_depends ${CMAKE_SOURCE_DIR}/tools/aoti-compile.py)
    if(AOTI_ALGORITHM_FILE)
        list(APPEND _aoti_depends ${_algorithm_file_abs})
    endif()
    list(APPEND _aoti_depends ${AOTI_DEPENDS})

    add_custom_command(
        OUTPUT
            ${_aoti_metadata_path}
            ${AOTI_OUTPUTS}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${AOTI_OUTPUT_DIR}
        COMMAND ${CMAKE_SOURCE_DIR}/tools/aoti-compile.py
            --algorithm-module ${_aoti_algorithm_module}
            ${_algorithm_class_args}
            --output-dir ${AOTI_OUTPUT_DIR}
            --metadata-path ${_aoti_metadata_path}
            --mode ${AOTI_MODE}
            ${_config_args}
        DEPENDS
            ${_aoti_depends}
        VERBATIM
    )

    add_custom_target(${AOTI_TARGET} ALL DEPENDS ${_aoti_metadata_path} ${AOTI_OUTPUTS})
endfunction()
