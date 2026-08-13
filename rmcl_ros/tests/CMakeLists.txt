
# GPU KERNELS
add_executable(rmcl_ros_tests_gpu_kernels rmcl_gpu_kernels.cpp)

target_link_libraries(rmcl_ros_tests_gpu_kernels
    rmcl_ros_cuda
)

# Whether the GPU library is built says only that the toolkit and rmagine's
# cuda component are installed, not that a device can be reached at test time.
# The probe answers that in a process that does not load rmagine, because
# rmagine creates its CUDA context while its library is being loaded and the
# test would abort before main() on a machine without a device.
add_executable(rmcl_ros_tests_gpu_probe gpu_device_probe.cpp)

target_include_directories(rmcl_ros_tests_gpu_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

if(USE_HIP)
    target_compile_definitions(rmcl_ros_tests_gpu_probe PRIVATE USE_HIP)
    target_link_libraries(rmcl_ros_tests_gpu_probe hip::host)
else(USE_HIP)
    target_include_directories(rmcl_ros_tests_gpu_probe PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(rmcl_ros_tests_gpu_probe ${CUDA_LIBRARIES})
endif(USE_HIP)

add_test(NAME rmcl_gpu_kernels COMMAND ${CMAKE_COMMAND}
    -DPROBE=$<TARGET_FILE:rmcl_ros_tests_gpu_probe>
    -DTEST=$<TARGET_FILE:rmcl_ros_tests_gpu_kernels>
    -P ${CMAKE_CURRENT_SOURCE_DIR}/run_gpu_test.cmake
)

set_tests_properties(rmcl_gpu_kernels PROPERTIES
    SKIP_REGULAR_EXPRESSION "no GPU device available"
)
