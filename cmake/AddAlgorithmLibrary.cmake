function(add_algorithm_library target)
    add_library(${target} SHARED ${ARGN})
    target_include_directories(
        ${target}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/inc
            ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(
        ${target}
        PUBLIC
            ndarray
            ${TORCH_LIBRARIES}
    )
endfunction()
