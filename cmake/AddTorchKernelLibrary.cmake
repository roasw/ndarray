function(add_torch_kernel_library target)
    if(ARGC LESS 2)
        message(FATAL_ERROR "add_torch_kernel_library requires at least one source")
    endif()

    add_library(${target} SHARED ${ARGN})
    target_include_directories(
        ${target}
        PUBLIC
            ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PUBLIC ${TORCH_LIBRARIES})
endfunction()
