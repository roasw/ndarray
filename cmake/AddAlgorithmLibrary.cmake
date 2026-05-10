function(add_algorithm_library target source)
    add_library(${target} SHARED ${source})
    target_include_directories(
        ${target}
        PUBLIC
            inc
            ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(
        ${target}
        PUBLIC
            ndarray
            ${TORCH_LIBRARIES}
    )
endfunction()
