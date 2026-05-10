function(add_torch_kernel_library target source)
    add_library(${target} SHARED ${source})
    target_include_directories(
        ${target}
        PUBLIC
            ${TORCH_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PUBLIC ${TORCH_LIBRARIES})
endfunction()
